from src.train import main as train_main
from src.test import main as test_main
from utils.setup_directories import create_project_structure
import argparse

def main():
    # Create necessary directories
    create_project_structure()
    
    parser = argparse.ArgumentParser(description='Scratch Detection')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                      help='train or test the model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_main()
    else:
        test_main()

if __name__ == '__main__':
    main()