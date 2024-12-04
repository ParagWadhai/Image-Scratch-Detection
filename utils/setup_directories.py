import os

def create_project_structure():
    """Create the required directory structure for the project."""
    directories = [
        'data/train/good',
        'data/train/bad',
        'data/test/good',
        'data/test/bad'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == '__main__':
    create_project_structure()