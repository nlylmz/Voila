import csv
import itertools
import os
import random
import shutil
from PIL import Image
import argparse

counter = 1

number = ["one", "two", "three", "four"]

object_types = ['female children', 'men', 'women', 'senior men', 'senior women', 'cats', 'dogs', 'foxes',
                    'hamsters', 'monkeys', "wolves",
                    "male children",
                    "bears",
                    "rabbits", ]

object_type = ['female child', 'man', 'woman', 'senior man', 'senior woman', 'cat', 'dog', 'fox', 'hamster',
               'monkey', "wolf",
               "male child",
               "bear",
               "rabbit", ]

actions = [
    "playing soccer",
    "driving a car",
    "ice-skating",
    "walking",
    "swimming",
    "jumping",
    "typing",
    "writing",
    "digging a hole",
    "carrying something",
    "reading",
    "running",
    "eating food",
]


def stable(property):
    # Generate all possible dual permutations of property values
    combinations = list(itertools.permutations(property, 2))
    new_list = [(x, x, y, y) for x, y in combinations]
    return new_list

def changed(property):
    combinations = list(itertools.permutations(property, 2))
    new_list = [(x, y, x, y) for x, y in combinations]
    return new_list


# -1, -2, -3, +1, +2, +3
def arithmetic():
    unseen_cases = [('five', 'six', 'five', 'six'), ('five', 'six', 'six', 'seven'),
                    ('six', 'seven', 'six', 'seven'), ('five', 'seven', 'five', 'seven'),
                    ('seven', 'five', 'seven', 'five'), ('seven', 'six', 'six', 'five'),
                    ('six', 'five', 'six', 'five'), ('seven', 'six', 'seven', 'six')]

    arithmetic_analogy_cases = [('one', 'two', 'one', 'two'), ('one', 'two', 'two', 'three'),
                              ('one', 'two', 'three', 'four'), ('two', 'three', 'two', 'three'),
                              ('two', 'three', 'three', 'four'), ('two', 'three', 'one', 'two'),
                              ('three', 'four', 'three', 'four'), ('three', 'four', 'one', 'two'),
                              ('three', 'four', 'two', 'three'),
                              ('one', 'three', 'one', 'three'), ('one', 'three', 'two', 'four'),
                              ('two', 'four', 'two', 'four'), ('two', 'four', 'one', 'three'),
                              ('one', 'four', 'one', 'four'),
                              ('two', 'one', 'two', 'one'), ('two', 'one', 'three', 'two'),
                              ('two', 'one', 'four', 'three'), ('three', 'two', 'three', 'two'),
                              ('three', 'two', 'four', 'three'), ('three', 'two', 'two', 'one'),
                              ('three', 'one', 'three', 'one'), ('three', 'one', 'four', 'two'),
                              ('four', 'two', 'four', 'two'), ('four', 'three', 'four', 'three'),
                              ('four', 'three', 'three', 'two'), ('four', 'three', 'two', 'one',),
                              ('four', 'one', 'four', 'one'), ('four', 'two', 'three', 'one',)]

    return arithmetic_analogy_cases

#Repeated analogies occur when all three properties are changed at the same time. Therefore, we remove cases like 1:2::1:2, where the same rule is applied again.
#When all three numbers are different, no repeated analogy is formed!
def nonrepetative_arithmetic():
    cases = [('one', 'two', 'two', 'three'), ('one', 'two', 'three', 'four'),
             ('two', 'three', 'three', 'four'), ('two', 'three', 'one', 'two'),
             ('three', 'four', 'one', 'two'), ('three', 'four', 'two', 'three'),
             ('one', 'three', 'two', 'four'), ('two', 'four', 'one', 'three'),
             ('two', 'one', 'three', 'two'), ('two', 'one', 'four', 'three'),
             ('three', 'two', 'four', 'three'), ('three', 'two', 'two', 'one'),
             ('four', 'three', 'three', 'two'), ('four', 'three', 'two', 'one',),
             ('three', 'one', 'four', 'two'), ('four', 'two', 'three', 'one',)]
    unseen_cases = [('five', 'six', 'six', 'seven'), ('seven', 'six', 'six', 'five')]
    return cases


# First three value will be different.
def distractor(property):
    combinations = list(itertools.permutations(property, 3))

    result_list = [(a, b, c, d) for (a, b, c) in combinations for d in property]
    return result_list


# First three value will be different and the substraction of first number to second number should be equal or bigger than the third image value.
# For example; 4:2 :: 1 ? N=-1. Total 6 distractor. First three numbers must be different.
def distractor_number():
    dist_cases_img4 = [('four', 'one', 'two'), ('four', 'one', 'three'), ('four', 'two', 'one'), ('four', 'three', 'one'),
                        ('three', 'one', 'two'), ('three', 'two', 'one')]

    result_list = [(a, b, c, d) for (a, b, c) in dist_cases_img4 for d in number if d != a and d != b and d != c]
    return result_list

# One property change at a time : action or number or object type
def one_property_change_analogy1(args):
    n = stable(number)
    o = stable(object_types)
    a = changed(actions)

    generate_analogy_questions(n, o, a, '1', args)

def one_property_change_analogy2(args):
    n = distractor_number()
    o = stable(object_types)
    a = changed(actions)

    generate_analogy_questions(n, o, a, '2', args)


def one_property_change_analogy3(args):
    n = stable(number)
    o = distractor(object_types)
    a = changed(actions)

    generate_analogy_questions(n, o, a, '3', args)

def one_property_change_analogy4(args):
    n = distractor_number()
    o = distractor(object_types)
    a = changed(actions)

    generate_analogy_questions(n, o, a, '4', args)

def one_property_change_analogy5(args):
    n = stable(number)
    o = changed(object_types)
    a = stable(actions)

    generate_analogy_questions(n, o, a, '5', args)

def one_property_change_analogy6(args):
    n = distractor_number()
    o = changed(object_types)
    a = stable(actions)

    generate_analogy_questions(n, o, a, '6', args)

def one_property_change_analogy7(args):
    n = stable(number)
    o = changed(object_types)
    a = distractor(actions)

    generate_analogy_questions(n, o, a, '7', args)

def one_property_change_analogy8(args):
    n = distractor_number()
    o = changed(object_types)
    a = distractor(actions)

    generate_analogy_questions(n, o, a, '8', args)

def one_property_change_analogy9(args):
    n = arithmetic()
    o = stable(object_types)
    a = stable(actions)

    generate_analogy_questions(n, o, a, '9', args)

def one_property_change_analogy10(args):
    n = arithmetic()
    o = distractor(object_types)
    a = stable(actions)

    generate_analogy_questions(n, o, a, '10', args)

def one_property_change_analogy11(args):
    n = arithmetic()
    o = stable(object_types)
    a = distractor(actions)

    generate_analogy_questions(n, o, a, '11', args)

def one_property_change_analogy12(args):
    n = arithmetic()
    o = distractor(object_types)
    a = distractor(actions)

    generate_analogy_questions(n, o, a, '12', args)

# Two properties change at a time : action or number or object type
def two_properties_change_analogy1(args):
    n = stable(number)
    o = changed(object_types)
    a = changed(actions)

    generate_analogy_questions(n, o, a, '13', args)

def two_properties_change_analogy2(args):
    n = distractor_number()
    o = changed(object_types)
    a = changed(actions)

    generate_analogy_questions(n, o, a, '14', args)

def two_properties_change_analogy3(args):
    n = arithmetic()
    o = changed(object_types)
    a = stable(actions)

    generate_analogy_questions(n, o, a, '15',args)

def two_properties_change_analogy4(args):
    n = arithmetic()
    o = changed(object_types)
    a = distractor(actions)

    generate_analogy_questions(n, o, a, '16', args)

def two_properties_change_analogy5(args):
    n = arithmetic()
    o = stable(object_types)
    a = changed(actions)

    generate_analogy_questions(n, o, a, '17', args)

def two_properties_change_analogy6(args):
    n = arithmetic()
    o = distractor(object_types)
    a = changed(actions)

    generate_analogy_questions(n, o, a, '18', args)

# Three properties change at a time : action or number or object type
def three_properties_change_analogy(args):
    n = nonrepetative_arithmetic()
    o = changed(object_types)
    a = changed(actions)

    generate_analogy_questions(n, o, a, '19', args)


def generate_analogy_questions(n, o ,a, r, args):
    
    if args.distraction == 'no':
         desired_count = args.count // 7
    elif args.distraction == 'yes':
         desired_count = args.count // 19
        
    # CSV file path
    csv_file_path = args.csv_output
    header = ['img1', 'img2', 'img3', 'img4', 'desc_img1', 'desc_img2', 'desc_img3', 'desc_img4', 'combined_descriptions',
                                'image_question', 'rule', 'relations']
    
    file_exists = os.path.isfile(csv_file_path)
    random_examples = select_randomly(n,o,a,desired_count)

    # Open CSV file in write mode
    with open(csv_file_path, mode='a', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(header)

        # Iterate through the selected examples
        for num, obj, action in random_examples:
            obj_list = list(obj)
            for i in range(len(obj_list)):
                # Check if the corresponding element in number is 'one'
                if num[i] == 'one':
                    # Find the index in the objects list
                    index = object_types.index(obj_list[i])
                    # Modify the corresponding element in obj_list
                    obj_list[i] = object_type[index]

            # Convert obj_list back to a tuple
            obj = tuple(obj_list)
            img1 = find_and_select_image(f"{num[0]}_{obj[0]}_{action[0]}_",0, args)
            img2 = find_and_select_image(f"{num[1]}_{obj[1]}_{action[1]}_",0, args)
            img3 = find_and_select_image(f"{num[2]}_{obj[2]}_{action[2]}_",0, args)
            desc_img1 = f"{num[0]} {obj[0]} {action[0]}"
            desc_img2 = f"{num[1]} {obj[1]} {action[1]}"
            desc_img3 = f"{num[2]} {obj[2]} {action[2]}"
            desc_img4, img4 = find_target_img(r, num[3], obj[3], action[3])
            combined_descriptions = f"Image 1:{desc_img1}. Image 2: {desc_img2}. Image 3: {desc_img3}"
            relations = find_relations(r, num, obj, action)
            if(args.collage == "yes"):
                image_question = combine_images(img1, img2, img3, args)
            else:
                image_question = ""
            rule = r
            if args.dataset == "testing":
                if img1 is not None and img2 is not None and img3 is not None:
                    row_data = [img1, img2, img3, img4, desc_img1, desc_img2, desc_img3, desc_img4, combined_descriptions, image_question, rule, relations]
                    writer.writerow(row_data)
            elif args.dataset == "training":
                img4 = find_and_select_image(f"{num[3]}_{obj[3]}_{action[3]}_", 0, args)
                if img1 is not None and img2 is not None and img3 is not None and img4 is not None:
                    if img1 == img4 or img2 == img4 or img3 == img4:
                        img4 = find_and_select_image(f"{num[3]}_{obj[3]}_{action[3]}_", 1, args)
                    row_data = [img1, img2, img3, img4, desc_img1, desc_img2, desc_img3, desc_img4, combined_descriptions,
                                image_question, rule, relations]
                    files = [img1, img2, img3, img4]
                    writer.writerow(row_data)
                    #Delete the saved image files in the folder to use other images equally.
                    delete_files(files)

def find_target_img(r , n, o, a):
    if r in ["3", "10", "18"]:
        desc_img4 = f"{n} any {a}"
        img4 = f"{n} {a}"
    elif r in ["2", "6", "14"]:
        desc_img4 = f"any {o} {a}"
        img4 = f"{o} {a}"
    elif r in ["7", "11", "16"]:
        desc_img4 = f"{n} {o} any"
        img4 = f"{n} {o}"
    elif r == "4":
        desc_img4 = f"any any {a}"
        img4 = f"{a}"
    elif r == "8":
        desc_img4 = f"any {o} any"
        img4 = f"{o} "
    elif r == "12":
        desc_img4 = f"{n} any any"
        img4 = f"{n}"
    else:
        desc_img4 = f"{n} {o} {a}"
        img4 = f"{n} {o} {a}"

    return desc_img4,img4

def find_relations(r, num, obj, action):
    if r in ["2", "11", "17"]:
        relations = f"Number is changed from {num[0]} to {num[1]}. " \
                    f"Action is changed from {action[0]} to {action[1]}. " \
                    f"Subject type remains constant {obj[0]}. "
    elif r in ["3", "7", "13"]:
        relations = f"Number remains constant {num[0]}. " \
                    f"Action is changed from {action[0]} to {action[1]}. " \
                    f"Subject type is changed from {obj[0]} to {obj[1]}. "
    elif r in ["6", "10", "15"]:
        relations = f"Number is changed from {num[0]} to {num[1]}. " \
                    f"Action remains constant {action[0]}. " \
                    f"Subject type is changed from {obj[0]} to {obj[1]}. "
    elif r == "1":
        relations = f"Number remains constant {num[0]}. " \
                    f"Action is changed from {action[0]} to {action[1]}. " \
                    f"Subject type remains constant {obj[0]}. "
    elif r == "5":
        relations = f"Number remains constant {num[0]}. " \
                    f"Action remains constant {action[0]}. " \
                    f"Subject type is changed from {obj[0]} to {obj[1]}. "
    elif r == "9":
        relations = f"Number is changed from {num[0]} to {num[1]}. " \
                    f"Action remains constant {action[0]}. " \
                    f"Subject type remains constant {obj[0]}. "
    else:
        relations = f"Number is changed from {num[0]} to {num[1]}. " \
                    f"Action is changed from {action[0]} to {action[1]}. " \
                    f"Subject type is changed from {obj[0]} to {obj[1]}. "

    return relations

def find_and_select_image(image_file, index, args):

    if args.dataset == "training":
        alt_folder_path = "Train_Images"
        folder_path = "Copy_Train_Images"
        if not os.path.exists(folder_path):
        # Copy the entire folder
            shutil.copytree(alt_folder_path, folder_path)
    elif args.dataset == "testing":
        folder_path = "Test_Images"

    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)
    # Filter files that start with "image" and have a common image file extension
    image_files = [file for file in all_files if file.lower().startswith(image_file)]

    #As there are 9 training images per category, distribution of the images are provided using copy folder.
    #If there is no matching image in the copy folder or only one image when index = 1.
    if not image_files or (index>0 and len(image_files)<2):
        alt_files = os.listdir(alt_folder_path)
        alt_image_files = [file for file in alt_files if file.lower().startswith(image_file)]

        #Copy the all find images from original folder to the copy folder
        if alt_image_files:
            for matched_file in alt_image_files:
                src_path = os.path.join(alt_folder_path, matched_file)
                dest_path = os.path.join(folder_path, matched_file)
                shutil.copy(src_path, dest_path)

            # Return the path to the first selected image
            return alt_image_files[0]
        else:
            print(f"No matching image files for " +{image_file} + " found in" + {folder_path} + " and " +{alt_folder_path})
            return None

    selected_image = image_files[index]
    return selected_image


def delete_files(file_names):
    folder_path = "Copy_Train_Images"
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        try:
            os.remove(file_path)
            #print(f"Deleted: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")

def delete_folder_if_exists():
    folder_path = "Copy_Train_Images"
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Delete the folder and its contents
        shutil.rmtree(folder_path)
        #print(f"Folder '{folder_path}' has been deleted.")


def select_randomly(numbers,object_types,actions,desired_count):
    from collections import Counter

    random.shuffle(numbers)
    random.shuffle(object_types)
    random.shuffle(actions)

    num_count = desired_count // len(numbers)
    action_count = desired_count // len(actions)
    object_count = desired_count // len(object_types)

    # Initialize counters to keep track of occurrences
    num_counter = Counter()
    action_counter = Counter()
    object_counter = Counter()

    # Initialize set to keep track of unique entries
    unique_entries = set()
    # Generate random selections with balanced distribution
    random_examples = []

    for _ in range(desired_count):
        while True:
            num_index = random.randrange(len(numbers))
            num = numbers[num_index]

            action_index = random.randrange(len(actions))
            action = actions[action_index]

            obj_index = random.randrange(len(object_types))
            obj = object_types[obj_index]

            if (num, obj, action) not in unique_entries:
                break

        # Update counters
        if num_counter[num_index] < num_count:
            num_counter[num_index] += 1

        if action_counter[action_index] < action_count:
            action_counter[action_index] += 1

        if object_counter[obj_index] < object_count:
            object_counter[obj_index] += 1

        unique_entries.add((num, obj, action))
        random_examples.append((num, obj, action))

    return random_examples

def combine_images(img1, img2, img3, args):
    global counter

    if args.dataset == "training":
        folder_path = "Train_Images/"
    elif args.dataset == "testing":
        folder_path = "Test_Images/"

    # Load the images
    images = [Image.open(folder_path+img1),
                Image.open(folder_path+img2),
                Image.open(folder_path+img3)]

    # Ensure the images have the same height before combining
    max_height = max(image.size[1] for image in images)

    # Resize images to have the same height
    images = [image.resize((int(image.width * max_height / image.height), max_height), Image.Resampling.LANCZOS) for
              image in images]

    # Create a new image with a width that is the sum of the images plus white frames
    frame_width = 50
    total_width = sum(image.size[0] for image in images) + (len(images) - 1) * frame_width
    combined_image = Image.new('RGB', (total_width, max_height), color='white')

    # Paste the images with white frames in between
    x_offset = 0
    for image in images:
        combined_image.paste(image, (x_offset, 0))
        x_offset += image.size[0] + frame_width

    # Determine the file name
    file_name = f'image_questions_{counter}.png'

    # Increment the counter for the next call
    counter += 1

    if not os.path.exists("Image_Questions"):
        os.makedirs("Image_Questions")

    # Save the combined image under Image_Questions folder
    combined_image.save(f'Image_Questions/{file_name}')

    # Return the file name for reference
    return file_name

def data_without_distraction():
    one_property_change_analogy1(args)
    one_property_change_analogy5(args)
    one_property_change_analogy9(args)

    two_properties_change_analogy1(args)
    two_properties_change_analogy3(args)
    two_properties_change_analogy5(args)

    three_properties_change_analogy(args)
    print('Analogy questions without distraction are ready')
    delete_folder_if_exists()

def data_with_distraction():
    one_property_change_analogy1(args)
    one_property_change_analogy2(args)
    one_property_change_analogy3(args)
    one_property_change_analogy4(args)
    one_property_change_analogy5(args)
    one_property_change_analogy6(args)
    one_property_change_analogy7(args)
    one_property_change_analogy8(args)
    one_property_change_analogy9(args)
    one_property_change_analogy10(args)
    one_property_change_analogy11(args)
    one_property_change_analogy12(args)

    two_properties_change_analogy1(args)
    two_properties_change_analogy2(args)
    two_properties_change_analogy3(args)
    two_properties_change_analogy4(args)
    two_properties_change_analogy5(args)
    two_properties_change_analogy6(args)

    three_properties_change_analogy(args)
    print('Analogy questions with distraction are ready')
    delete_folder_if_exists()


def main(args):

    if args.distraction == 'no':
        data_without_distraction()
        args.count = args.count // 7
    elif args.distraction == 'yes':
        data_with_distraction()
        args.count = args.count // 19


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create training or test dataset using generated images")
    parser.add_argument("--csv_output", type=str, required=True, help="Path to save the output CSV file")
    parser.add_argument("--dataset", type=str, choices=["training", "testing"], required=True, help="Choose one of the following options for creating the dataset: training and testing")
    parser.add_argument("--distraction", type=str, choices=["no", "yes"], required=True, help="Choose the option for distraction: yes, no")
    parser.add_argument("--count", type=int, required=True, help="The total number of questions to generate. For equal distribution of each rule please write the number divided by count of rules(7 or 19)")
    parser.add_argument("--collage", type=str, choices=["no", "yes"], required=True, help="Choose the option to create the image collages for questions: yes, no")
    args = parser.parse_args()
    main(args)











