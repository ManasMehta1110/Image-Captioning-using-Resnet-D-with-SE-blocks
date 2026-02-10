from utils import create_input_files
import os

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))

    create_input_files(
        dataset='coco',
        karpathy_json_path=os.path.join(
            base_dir, 'data', 'coco', 'annotations', 'dataset_coco.json'
        ),
        image_folder=os.path.join(
            base_dir, 'data', 'coco', 'images'
        ),
        captions_per_image=5,
        min_word_freq=5,
        output_folder=os.path.join(
            base_dir, 'data', 'coco', 'processed'
        ),
        max_len=50
    )
