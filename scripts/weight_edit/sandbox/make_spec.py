"""Print the forward-pass server specification (module docstring + LIMITS) without importing the server."""
import ast
import sys

t = ast.parse(open(sys.argv[1]).read())
lim = next(n for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LIMITS")
print("# Forward-pass server specification (copied from the server source at launch)\n\n```\n"
      + ast.get_docstring(t) + "\n```\n\nLIMITS = " + ast.unparse(lim.value))
