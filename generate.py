"""
Generate deformation dataset for PartNet-Mobility using forward kinematics.
See arguments for more details.
"""

import os
import json
import random
import argparse
import subprocess
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import trimesh
import numpy as np
from tqdm import tqdm
from urchin import URDF
from termcolor import colored
import xml.etree.ElementTree as ET


ls = os.listdir
join = os.path.join
exists = os.path.exists
dname = os.path.dirname
bname = os.path.basename
mkdir = lambda d: os.makedirs(d, exist_ok=True)


def merge_meshes(meshes):
    vertices_all = []
    faces_all = []
    vertex_offset = 0

    for mesh in meshes:
        vertices = mesh.vertices  # shape: (N, 3)
        faces = mesh.faces        # shape: (M, 3)

        vertices_all.append(vertices)
        faces_all.append(faces + vertex_offset)

        vertex_offset += vertices.shape[0]

    merged_vertices = np.vstack(vertices_all)
    merged_faces = np.vstack(faces_all)

    return merged_vertices, merged_faces


class RobotInstance:
    """Manage a single instance of a robot using urchin.
    """
    
    def __init__(
        self,
        urdf_path: str, 
        lazy_load: bool = False,
    ):
        
        self.urdf_path = urdf_path
        self.lazy_load = lazy_load
        
        self.check_and_load(urdf_path, lazy_load=lazy_load)
        
    def fill_fields(self):
        
        urdf_path = self.urdf_path
        urdf_path_new = urdf_path.replace('mobility.urdf', 'mobility_fix.urdf')
        self.urdf_path = urdf_path_new
        
        if exists(urdf_path_new):
            return urdf_path_new
        
        # Fill in missing fields, hardcoded for sapien dataset
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for joint in root.findall("joint"):
            limit = joint.find("limit")
            if limit is not None:
                if "effort" not in limit.attrib:
                    limit.set("effort", "30")  # These are just random values
                if "velocity" not in limit.attrib:
                    limit.set("velocity", "1.0")  # Also these
        tree.write(urdf_path_new, encoding="utf-8", xml_declaration=True)        
        
        return urdf_path_new
    
    def check_and_load(self, urdf_path: str, lazy_load: bool):
        if not exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        # Fill missing "effort" and "velocity" field for all non-fixed joints (to be compatible with urchin)
        urdf_path = self.fill_fields()
        
        if lazy_load:
            self.robot = URDF.load(urdf_path, lazy_load_meshes=True)
        else:
            self.robot = URDF.load(urdf_path)
            
        if len(self.robot.actuated_joints) == 0:
            raise ValueError(f"Robot {urdf_path} has no actuated joints.")

    def gen_random_joint_values(self, num: int, seed: int = 0) -> List[Dict[str, float]]:
        
        np.random.seed(seed)
        random_values = np.random.rand(num, len(self.robot.actuated_joints))
        
        joint_values = []
        for i in range(num):
            joint_values_i = {}
            
            for j, joint in enumerate(self.robot.actuated_joints):  # Only modify non-fixed joints
                
                if joint.limit is None:  # Continuous joint
                    lower, upper = -2 * np.pi, 2 * np.pi
                else:
                    lower, upper = joint.limit.lower, joint.limit.upper
                
                if lower is not None and upper is not None:
                    value = float(random_values[i, j] * (upper - lower) + lower)
                    joint_values_i[joint.name] = value
                        
            joint_values.append(joint_values_i)
            
        return joint_values
    
    def export_mesh(self, output_path: str, joint_values: Dict[str, float]):
        """Export the modified robot mesh as a single OBJ file."""
        
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # Compute FK for the new joint values
        fk_transforms = self.robot.link_fk(cfg=joint_values)

        for link in self.robot.links:
            
            if not link.visuals:
                continue
        
            for visual in link.visuals:
                if visual.geometry.mesh is not None:
                    
                    if len(visual.geometry.mesh.meshes) == 0:
                        raise ValueError(f"Expected at least 1 mesh, but found none.")
                    
                    vertices, faces = merge_meshes(visual.geometry.mesh.meshes)
                    
                    local_transform = visual.origin  
                    global_transform = fk_transforms[link]  
                    final_transform = global_transform @ local_transform
                    transformed_vertices = (final_transform[:3, :3] @ vertices.T).T + final_transform[:3, 3]

                    all_vertices.extend(transformed_vertices)
                    all_faces.extend(faces + vertex_offset)
                    vertex_offset += len(vertices)

        final_mesh = trimesh.Trimesh(vertices=np.array(all_vertices), faces=np.array(all_faces))
        final_mesh.export(output_path)
        
    def export_joints(self, output_path: str, joint_values: Dict[str, float]):
        with open(output_path, 'w') as f:
            json.dump(joint_values, f)
        
    def generate_deformations(
        self, 
        export_path: str, 
        num_frames: int, 
        seed: int = 0,
    ):
        joint_values = self.gen_random_joint_values(num_frames, seed=seed)
        
        for f in tqdm(range(num_frames), desc=f'Generating @ {export_path}'):
            mesh_path = join(export_path, f"mesh_{f:04d}.obj")
            joint_path = join(export_path, f"joints_{f:04d}.json")
            
            self.export_mesh(mesh_path, joint_values[f])
            self.export_joints(joint_path, joint_values[f])


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def verify_partnet_mobility(root_dir: str):
    """
    Fill missing "effort" and "velocity" field for all non-fixed joints (to be compatible with urchin)
    Verify all instances of partnet mobility dataset
    """
    
    valid_files = sorted([join(root_dir, d, 'mobility.urdf') for d in ls(root_dir) if exists(join(root_dir, d, 'mobility.urdf'))])
    print(f'Found {len(valid_files)} valid files in current root directory')
    
    invalid_files = []
    for urdf_path in tqdm(valid_files):        
        try:
            _ = RobotInstance(urdf_path, lazy_load=True)  # Skip loading mesh
        except Exception as e:
            print(f"Error loading URDF {urdf_path}: {e}")
            invalid_files.append(urdf_path)
    
    print(f'Found {len(invalid_files)} invalid files in current root directory:')
    for i, invalid_file in enumerate(invalid_files):
        print(f'{i + 1}: {invalid_file}')
    
    print(colored('Verification done.', 'green'))
    
    
def get_categories(root_dir: str):
    valid_dirs = sorted([join(root_dir, d) for d in ls(root_dir) if exists(join(root_dir, d, 'mobility.urdf'))])
    meta_paths = [join(d, 'meta.json') for d in valid_dirs]
    cats = {}
    for meta_path in tqdm(meta_paths):
        meta = read_json(meta_path)
        if meta['model_cat'] not in cats.keys():
            cats[meta['model_cat']] = 1
        else:
            cats[meta['model_cat']] += 1
    return cats
    
    
def show_class_partnet_mobility(root_dir):
    cats = get_categories(root_dir)
    print(f'Found {len(cats)} categories, {sum(cats.values())} instances in current root directory:')
    for k, v in cats.items():
        print(f'Class {k}: {v}')
    print(colored('Show categories done.', 'green'))
    
    
def get_existing_categories(root_dir: str, cats: List[str]) -> List[str]:
    root_metas = sorted([f for f in ls(root_dir) if '.json' in f])
    existing_cats = []
    for root_meta in root_metas:
        if 'all' in root_meta: 
            return cats  # The entire dataset already exists
        for cat in cats:
            if cat in root_meta and cat not in existing_cats:
                existing_cats.append(cat)
    return existing_cats


def generate_partnet_mobility(
    root_dir: str, 
    inc_cat: List[str],
    exc_cat: List[str],
    num_frames: int,
    overwrite: bool,
    seed: int = 0,
):
    """
    Generate deformation dataset for PartNet-Mobility
    """
    
    cats = get_categories(root_dir)
    if len(inc_cat) > 0:
        cats = [c for c in cats if c in inc_cat]
    if len(exc_cat) > 0:
        cats = [c for c in cats if not c in exc_cat]
    
    title = 'all' 
    if len(inc_cat) > 0 or len(exc_cat) > 0:
        title = [''.join(cats) for cats in sorted(cats)][0]
    
    valid_files = sorted([join(root_dir, d, 'mobility.urdf') for d in ls(root_dir) if exists(join(root_dir, d, 'mobility.urdf'))])
    
    if not overwrite:
        print('Searching for previously generated categories...')
        existing_cats = get_existing_categories(root_dir, cats)
        print('Found existing categories:', existing_cats)
    
    invalid_files = []
    meta_info = []
    
    for urdf_path in tqdm(valid_files):
    
        # This is pretty slow
        meta_path = urdf_path.replace('mobility.urdf', 'meta.json')
        meta = read_json(meta_path)
        
        if meta['model_cat'] not in cats:
            continue
        
        if not overwrite and meta['model_cat'] in existing_cats:
            # Link meta
            meta_info.append(
                {
                    'category': meta['model_cat'],
                    'model_id': meta['model_id'],
                    'anno_id': meta['anno_id'],
                    'user_id': meta['user_id'],
                    'time_in_sec': meta['time_in_sec'],
                    'version': meta['version'],
                    'path': dname(urdf_path),
                }
            )
            continue
        
        try:
            robot = RobotInstance(urdf_path, lazy_load=False)
            export_dir = urdf_path.replace('mobility.urdf', 'deformations')
            mkdir(export_dir)
            robot.generate_deformations(export_dir, num_frames, seed)
            del robot

        except Exception as e:
            print(f"Error generating URDF for {urdf_path}: {e}")
            invalid_files.append(urdf_path)
            continue
        
        # Write meta
        meta_info.append(
            {
                'category': meta['model_cat'],
                'model_id': meta['model_id'],
                'anno_id': meta['anno_id'],
                'user_id': meta['user_id'],
                'time_in_sec': meta['time_in_sec'],
                'version': meta['version'],
                'path': dname(urdf_path),
            }
        )
    
    with open(join(root_dir, 'meta_' + title + '.json'), 'w') as f:
        json.dump(meta_info, f, indent=4)
    
    if len(invalid_files) > 0:
        print(f'Unable to generate for invalid files:')
        for i, invalid_file in enumerate(invalid_files):
            print(f'{i + 1}: {invalid_file}')
    
    print(colored(f'Generation done ({len(meta_info)} Sequences). Congrats.', 'green'))
    
    
def run_manifold(
    manifold_exe_path: str,
    input_path: str,
    output_path: str,
    simplify_exe_path: str = '',
    num_face: int = 5000,
):
    subprocess.run([manifold_exe_path, input_path, output_path, '20000'])
    if len(simplify_exe_path) > 0:
        subprocess.run([simplify_exe_path, '-i', output_path, '-o', output_path.replace('.obj', '_simplified.obj'), '-f', str(num_face)])


def run_manifold_multi(
    manifold_exe_path: str,
    input_path: List[str],
    output_path: List[str],
    simplify_exe_path: str = '',
    num_face: int = 5000,
    max_workers: int = 8,
):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_manifold, 
                manifold_exe_path, 
                input_path[i], 
                output_path[i], 
                simplify_exe_path,
                num_face,
            ) for i in range(len(input_path))
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing manifolds..."):
            pass


def manifold_partnet_mobility(
    root_dir: str, 
    inc_cat: List[str],
    exc_cat: List[str],
    manifold_exe_path: str,
    simplify_exe_path: str,
    num_faces: int = 5000,
    max_workers: int = 8,
):
    cats = get_categories(root_dir)
    if len(inc_cat) > 0:
        cats = [c for c in cats if c in inc_cat]
    if len(exc_cat) > 0:
        cats = [c for c in cats if not c in exc_cat]
    
    title = 'all' 
    if len(inc_cat) > 0 or len(exc_cat) > 0:
        title = [''.join(cats) for cats in sorted(cats)][0]
        
    meta_path = join(root_dir, 'meta_' + title + '.json')
    meta = read_json(meta_path)

    seq_dirs = [d['path'] for d in meta]
    seq_paths = [join(d, 'deformations') for d in seq_dirs]
    print(f'Found {len(seq_paths)} sequences/instances.')
    
    input_paths = []
    output_paths = []
    for seq_path in seq_paths:
        frame_paths = sorted([join(seq_path, f) for f in ls(seq_path) \
            if ('.obj' in f) and ('manifold' not in f) and ('simplified' not in f)])
        input_paths.extend(frame_paths)
        output_paths.extend([f.replace('.obj', '_manifold.obj') for f in frame_paths])
    
    run_manifold_multi(
        manifold_exe_path=manifold_exe_path,
        input_path=input_paths,
        output_path=output_paths,
        simplify_exe_path=simplify_exe_path,
        num_face=num_faces,
        max_workers=max_workers,
    )
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='2 stage pipeline for loading a SAPIEN URDF files and generating deformation dataset')
    parser.add_argument('-r', '--root', type=str, required=True, help='Path to the root directory of the PartNet-Mobility dataset.')
    
    # Different modes
    parser.add_argument('-g', '--gen', action='store_true', default=False, help='Perform generation stage on the entire PartNet-Mobility dataset, must run after "verify" stage.')
    parser.add_argument('-s', '--show', action='store_true', default=False, help='Show all categories.')
    parser.add_argument('-v', '--verify', action='store_true', default=False, help='Perform verfiy stage on the entire PartNet-Mobility dataset.')
    parser.add_argument('-m', '--manifold', action='store_true', default=False, help='Run manifold algorithm on generated classes, must be run after "gen" stage.')

    parser.add_argument('-i', '--include', type=str, default='', help='A list of categories to generate.')
    parser.add_argument('-e', '--exclude', type=str, default='', help='A list of categories to exclude from generation.')
    parser.add_argument('-o', '--overwrite', action='store_true', default=False, help='Overwrite previously generated dataset.')
    
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
    parser.add_argument('--mpath', type=str, default='', help='Path to the manifold generation executable, only required in "manifold" mode.')  # /afs/cs.stanford.edu/u/alexhe/generic_rigging/ImplicitRigging/Manifold/build/manifold
    parser.add_argument('--spath', type=str, default='', help='Path to the mesh simplification executable, optional, only used in "manifold" mode.')  # /afs/cs.stanford.edu/u/alexhe/generic_rigging/ImplicitRigging/Manifold/build/simplify
    parser.add_argument('-f', '--frame', type=int, default=100, help='Number of frames to generate for each instance.')

    args = parser.parse_args()
    
    assert (int(args.show) + int(args.verify) + int(args.gen) + int(args.manifold)) == 1, \
        'Only one of "verify", "show", "manifold" and "gen" can be set to True at a time.'
    assert not (len(args.include.split()) and len(args.exclude.split())), \
        'Only one of "include" and "exclude" can be set at a time.'

    if args.verify:
        verify_partnet_mobility(args.root)
        
    elif args.gen:
        generate_partnet_mobility(
            root_dir=args.root, 
            inc_cat=args.include.split(), 
            exc_cat=args.exclude.split(),
            num_frames=args.frame,
            overwrite=args.overwrite,
            seed=args.seed,
        )
        
    elif args.show:
        show_class_partnet_mobility(args.root)
        
    elif args.manifold:
        manifold_partnet_mobility(
            root_dir=args.root,
            inc_cat=args.include.split(),
            exc_cat=args.exclude.split(),
            manifold_exe_path=args.mpath,
            simplify_exe_path=args.spath,
            num_faces=5000,
            max_workers=8,
        )
    
    else:
        raise ValueError('One of "verify", "show" and "gen" must be set to True.')
