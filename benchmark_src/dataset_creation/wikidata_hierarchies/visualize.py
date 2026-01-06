from treelib import Tree
from pathlib import Path

SAVE_DIR = Path("data/wikidata_genres")

def create_tree(dictionary, tree=None, parent=None):
    if tree is None:
        tree = Tree()
        tree.create_node("Root", "root")  # Create a root node if needed
        parent = "root"
    for key, value in dictionary.items():
        tree.create_node(str(key), str(key), parent=parent)
        if isinstance(value, dict):
            create_tree(value, tree, str(key))
    return tree

def visualize_hierarchy(nested_dict):
    tree = create_tree(nested_dict)
    #tree.show()
    # TODO: save_dir as a parameter!
    save_path = SAVE_DIR / 'hierarchy_tree.txt'
    if save_path.exists():
        save_path.unlink()
    tree.save2file(save_path)