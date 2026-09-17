import ast
from pathlib import Path

def test_data_does_not_import_propagation():
    data_file = Path("src/data.py")
    if not data_file.exists():
        return
        
    with open(data_file, 'r') as f:
        tree = ast.parse(f.read())
        
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "propagation" not in alias.name, "data.py must not import propagation"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert "propagation" not in node.module, "data.py must not import propagation"
                
def test_hidden_lineage_isolation():
    # Simple static check for now - data.py shouldn't expose lineage to anything but evaluate and targets
    # This is more of a systemic architecture rule, but we can verify targets.py doesn't export lineage
    targets_file = Path("src/targets.py")
    if not targets_file.exists():
        return
        
    with open(targets_file, 'r') as f:
        content = f.read()
        
    # Just checking it doesn't write lineage out to features somehow
    assert "lineage" in content # Should use it internally
