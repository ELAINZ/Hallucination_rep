import os
import cv2
import random
import copy
import math
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from easydict import EasyDict
import pickle

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data import get_metadata, get_dataset, fix_legacy_dict, get_synthetic_dataset
#from data import get_metadata, get_dataset, fix_legacy_dict, get_synthetic_dataset
import PIL
import wandb
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

def calculate_mse(image1, image2):

    return np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
def get_generation_number(file_name):
    return int(file_name.split('_')[1])
def main():
    parser = argparse.ArgumentParser("Minimal implementation of diffusion models")
 
    parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
    parser.add_argument('--model_dir', default='0', type=str, help='number to indicate which experiment it is')
    parser.add_argument(
        "--log_results",
        action="store_true",
        default=False)
    args = parser.parse_args()

    args.store_name = '_'.join(['analysis',args.model_dir,args.exp_str])
    save_dir = os.path.join("trained_models", args.store_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if args.log_results:
        wandb.init(project="synthetic",
                                    entity="neurips", name=args.store_name)
        wandb.config.update(args)
        wandb.run.log_code(".")
    #real_metadata = np.load("simple-shapes/meta_data.npz", allow_pickle=True)
    #loaded_data = np.load(real_metadata, allow_pickle=True)
    #overall_figure_meta_data = real_metadata['arr_0'].item()
    #analysis(overall_figure_meta_data, -1, args)
    #assert 0
    args.model_dir = os.path.join("trained_models", args.model_dir)
    filtered_npz = [i for i in os.listdir(args.model_dir) if i.endswith(".npz")]
    sorted_filtered_npz = sorted(filtered_npz, key=get_generation_number)
    mapped = {0:"triangle", 1:"square", 2:"pentagon"}
    real_images = []
    real_images_dir = "simple-shapes/"
    for filename in os.listdir(real_images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(real_images_dir, filename))
            real_images.append(np.asarray(img))
    min_mse_values = []
    for path in sorted_filtered_npz:
        
        data = np.load(os.path.join(args.model_dir, path))
        gen = get_generation_number(path)
        print(path, gen)
        if gen > 10:
           continue
        generated_images = data['X']
        count_dict = {0:0, 1:0, 2:0}
        image_shape_dict = {}
        pixel_values = []

        for idx, image_array in enumerate(generated_images):
            #img_ = Image.fromarray(img.squeeze())
            image_shape_dict[idx] = []
            # print(image_array.shape)

            column_width = 21
            original_image_array = image_array
            #thresholded_original = original_image_array[original_image_array > 200] = 255
            pixel_values.extend(original_image_array.ravel())
            image_array = image_array[1:64, 1:64]
            image_array[image_array > 200] = 255
            columns = [image_array[:, i * column_width:(i + 1) * column_width] for i in range(3)]
            # print(columns[0].shape)
            
            # Count the number of white pixels (pixel value 255) in each column
            pixel_counts = [np.count_nonzero(column == 255) for column in columns]

            # Print the pixel counts
            for i, count in enumerate(pixel_counts):
                num_ = count/65
                rounded_num = round(num_)
                count_dict[i] += rounded_num
                if 0:#rounded_num > 1:
                    print(original_image_array.shape)
                    pil_image = Image.fromarray(original_image_array.reshape(64, 64).astype(np.uint8))
                    if not os.path.exists(os.path.join(save_dir, f"gen{gen}-2{mapped[i]}")):
                        os.mkdir(os.path.join(save_dir, f"gen{gen}-2{mapped[i]}"))
                    pil_image.save(os.path.join(save_dir, f"gen{gen}-2{mapped[i]}",f"{idx}.png"))
                    if args.log_results:
                        wandb.log({f"gen_{gen}-{rounded_num}-{mapped[i]}":wandb.Image(pil_image, caption=f"Gen{gen}-{rounded_num}-{mapped[i]}-idx{idx}")})
                if rounded_num!=0:
                    image_shape_dict[idx].append([mapped[i], rounded_num])
             
        #for idx in image_shape_dict:
        plt.hist(pixel_values, bins=256, range=(0, 256), density=True, alpha=0.5, label=f"Gen-{gen}")
        if args.log_results:
            #plt.ylabel("some interesting numbers")
            wandb.log({f"gen-{gen}-pixel-intensity": plt})
        for idx,image_data in enumerate(image_shape_dict.values()):
            print(idx) 
            shapes = [item[0] for item in image_data]
            counts = [item[1] for item in image_data]
            if len(counts)>0 and max(counts) > 1:
                name = "_".join([f"{count}_{shape}" for shape, count in zip(shapes,counts)]) 
                if args.log_results:
                    
                    pil_image = Image.fromarray(generated_images[idx].reshape(64, 64).astype(np.uint8))
                    if not os.path.exists(os.path.join(save_dir, f"gen{gen}-{name}")):
                        os.mkdir(os.path.join(save_dir, f"gen{gen}-{name}"))
                    pil_image.save(os.path.join(save_dir, f"gen{gen}-{name}",f"{idx}.png"))
                    if args.log_results:
                        wandb.log({f"gen_{gen}-{name}":wandb.Image(pil_image, caption=f"Gen{gen}-{name}-idx{idx}")})
            else:
                 
                thresholded_array = np.copy(generated_images[idx])
                thresholded_array[thresholded_array > 200] = 255
                thresholded_array[thresholded_array <= 200] = 0
                min_mse = min([calculate_mse(thresholded_array, real_image) for real_image in real_images])
                min_mse_values.append((idx, min_mse)) 
        min_mse_values.sort(key=lambda x: x[1])
        print(min_mse_values)
        top_5000_synthetic_images = min_mse_values[:5000]
        copy_gen_images = np.copy(generated_images)
        copy_gen_images[copy_gen_images > 200] = 255
        copy_gen_images[copy_gen_images <= 200] = 0
        indices = [j[0] for j in min_mse_values]
        filtered_images = copy_gen_images[indices]
        print(len(filtered_images))
        data_dict = {'X': [], 'Y': []}
        data_dict['X'] = filtered_images
        data_dict['Y'] = np.zeros((len(filtered_images)))
        np.savez(os.path.join(args.model_dir,f"filtered_gen_{gen}_generated_data_epoch_{args.epochs}-timesteps_{args.diffusion_steps}_sampling-steps_{args.sampling_steps}.npz"), **data_dict)
        analysis(image_shape_dict, gen, args)            
        
def analysis(image_shape_dict, gen, args):
    no_shapes_count = 0
    triangle_only_count = 0
    square_only_count = 0
    pentagon_only_count = 0
    triangle_square_count = 0
    triangle_pentagon_count = 0
    square_pentagon_count = 0
    all_shapes_count = 0
    two_triangle_count = 0

    shape_combinations_count = {}
    shapes = ['triangle', 'square', 'pentagon']
    counts = ['1', '2', '3']

    for shape1 in shapes:
        for shape2 in shapes:
            if shape1==shape2:
                continue
            for count1 in counts:
                shape_combinations_count[f'{count1}{shape1}'] = 0
                for count2 in counts:
                    if count1 or count2:
                        key = f'{count1}{shape1}_{count2}{shape2}'
                        shape_combinations_count[key] = 0

    # Add the 'all_shapes' and 'no_shapes' keys to the dictionary

    # Print the dictionary
    #print(shape_combinations_count)
    for image_data in image_shape_dict.values():
        if not image_data:
            continue
        shapes = [item[0] for item in image_data]
        counts = [item[1] for item in image_data]
        #counts = [1 for i in counts]
        counts = [min(i,3) for i in counts]

        if not shapes:
            no_shapes_count += 1
        elif 'triangle' in shapes and 'square' not in shapes and 'pentagon' not in shapes:
            triangle_only_count += 1
            shape_combinations_count[f'{counts[0]}triangle'] += 1
        elif 'square' in shapes and 'triangle' not in shapes and 'pentagon' not in shapes:
            square_only_count += 1
            shape_combinations_count[f'{counts[0]}square'] += 1
        elif 'pentagon' in shapes and 'triangle' not in shapes and 'square' not in shapes:
            pentagon_only_count += 1
            shape_combinations_count[f'{counts[0]}pentagon'] += 1
        elif 'triangle' in shapes and 'square' in shapes and 'pentagon' not in shapes:
            triangle_square_count += 1
            shape_combinations_count[f'{counts[0]}triangle_{counts[1]}square'] += 1
        elif 'triangle' in shapes and 'pentagon' in shapes and 'square' not in shapes:
            triangle_pentagon_count += 1
            shape_combinations_count[f'{counts[0]}triangle_{counts[1]}pentagon'] += 1
        elif 'square' in shapes and 'pentagon' in shapes and 'triangle' not in shapes:
            square_pentagon_count += 1
            shape_combinations_count[f'{counts[0]}square_{counts[1]}pentagon'] += 1
        elif all(shape in shapes for shape in ['triangle', 'square', 'pentagon']):
            all_shapes_count += 1

    # Print the counts
    print(f'No Shapes: {no_shapes_count}')
    print(f'Triangle Only: {triangle_only_count}')
    print(f'Square Only: {square_only_count}')
    print(f'Pentagon Only: {pentagon_only_count}')
    print(f'Triangle and Square: {triangle_square_count}')
    print(f'Triangle and Pentagon: {triangle_pentagon_count}')
    print(f'Square and Pentagon: {square_pentagon_count}')
    print(f'Triangle, Square, and Pentagon: {all_shapes_count}')
    print("Shape Combinations Count", shape_combinations_count)
    if args.log_results and args.local_rank==0:
        #wandb.log({'gen':gen, *shape_combinations_count})
        #for key, count in shape_combinations_count.items():
        #    wandb.log({key: count, "gen":gen})

        wandb.log({'gen':gen, "no_shapes":no_shapes_count, "triangle_only":triangle_only_count, "square_only":square_only_count, "pentagon_only":pentagon_only_count,
              "triangle_square":triangle_square_count, "triangle_pentagon":triangle_pentagon_count, "square_pentagon":square_pentagon_count, "all_shapes":all_shapes_count})


def filter_gen_images(image_shape_dict):
    # assert False
    filtered_indices = []
    for idx in image_shape_dict:
        image_data = image_shape_dict[idx]
        if not image_data:
            continue
        shapes = [item[0] for item in image_data]
        counts = [item[1] for item in image_data]
        if max(counts) <=1:
            filtered_indices.append(idx)
    return filtered_indices

def filter_gen_images_cv(image_shape_dict):
    filtered_indices = []
    for idx in image_shape_dict:
        image_data = image_shape_dict[idx]
        if not image_data:
            continue
        if '2' in image_data or '3' in image_data:
            continue
        filtered_indices.append(idx)
    return filtered_indices

def filter_gen_images_new(image_shape_dict, img_shape_dict_cv):
    filtered_indices = []
    for idx in image_shape_dict:
        image_data = image_shape_dict[idx]
        if not image_data:
            continue
        shapes = [item[0] for item in image_data]
        counts = [item[1] for item in image_data]
        if max(counts) > 1:
            continue
            #filtered_indices.append(idx)
        name = "_".join([f"{count}_{shape}" for shape, count in zip(shapes,counts)]) 
        if idx not in img_shape_dict_cv:
            print(idx)
        name_cv = img_shape_dict_cv[idx]
        if name==name_cv:
            filtered_indices.append(idx)
    return filtered_indices


def classify_simple_shapes(generated_images, gen, args):
    image_shape_dict = {}
    count_types = {}
    for idx in range(len(generated_images)):
        img = generated_images[idx]
        label = classify_image(img)
        shapes = [item[0] for item in label]
        counts = [item[1] for item in label]
        name = "_".join([f"{count}_{shape}" for shape, count in zip(shapes,counts)]) 
        image_shape_dict[idx] = label
        if name not in count_types:
            count_types[name] = 0
        count_types[name] += 1
       
        if 0:#(len(counts)>0 and max(counts) > 1) or (random.random() < 0.2):
            #pil_image = Image.fromarray(np.asarray(img).astype(np.uint8))
         
            #print(np.array(img).max(), np.array(img).min()) 
            if args.local_rank==0 and args.log_results:
         
                wandb.log({f"gen_{gen}-{name}":wandb.Image(pil_image, caption=f"Gen{gen}-{name}-idx{j}")})
    #import pdb; pdb.set_trace()
    # print(image_shape_dict)
    analysis(image_shape_dict, gen, args)
    if args.log_results and args.local_rank==0:
        for key in count_types:
            wandb.log({key:count_types[key], "gen":gen})
    return image_shape_dict

def classify_shape_cv(contour):
    # Approximate the contour to get the number of vertices
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    # Classify based on the number of vertices
    if num_vertices == 4:
        return "square"
    elif num_vertices == 3:
        return "triangle"
    elif num_vertices == 5:
            return "pentagon"
    else:
        return "unknown"

def classify_simple_shapes_cv(generated_images, gen, args):
    image_shape_dict = {}
    count_types = {}
    for idx in range(len(generated_images)):
        img = generated_images[idx].squeeze()
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        # _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
    )

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count occurrences of each shape
        # shape_counts = {"Triangle": 0,"Square": 0, "Pentagon": 0}
        shape_counts = {}

        for contour in contours:
            shape_label = classify_shape_cv(contour)
            if shape_label not in shape_counts:
                shape_counts[shape_label] = 0
            shape_counts[shape_label] += 1

        #name = "_".join([f"{count}_{shape}" for shape, count in shape_counts.items()])
        temp = []
        for s in ['triangle', 'square', 'pentagon', 'unknown']:
            if s not in shape_counts:
                continue
            if shape_counts[s]!=0:
                temp.extend([str(shape_counts[s]),s])
        name = '_'.join(temp)
        image_shape_dict[idx] = name
        if name not in count_types:
            count_types[name] = 0
        count_types[name] += 1
    for key in count_types:
        print(key, count_types[key])
    if args.log_results and args.local_rank==0:
        for key in count_types:
            wandb.log({key:count_types[key], "gen":gen})
    #print(image_shape_dict)
    return image_shape_dict

def classify_image(image_array):
            final = []
            mapped = {0:"triangle", 1:"square", 2:"pentagon"}
            column_width = 21
            original_image_array = image_array
            #thresholded_original = original_image_array[original_image_array > 200] = 255
            #pixel_values.extend(original_image_array.ravel())
            image_array = image_array[1:64, 1:64]
            image_array[image_array > 180] = 255
            columns = [image_array[:, i * column_width:(i + 1) * column_width] for i in range(3)]
            # print(columns[0].shape)
            
            # Count the number of white pixels (pixel value 255) in each column
            pixel_counts = [np.count_nonzero(column == 255) for column in columns]

            count_dict = {0:0, 1:0, 2:0}
            # Print the pixel counts
            for i, count in enumerate(pixel_counts):
                num_ = count/65
                rounded_num = round(num_)
                count_dict[i] += rounded_num
                if rounded_num!=0:
                    final.append([mapped[i], rounded_num])
            return final  


	 
if __name__ == "__main__":
    main()
