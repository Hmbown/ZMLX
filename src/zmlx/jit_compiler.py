from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

from .metal import kernel as metal_kernel
from .msl import DEFAULT_HEADER


class MetalExprVisitor(ast.NodeVisitor):
    def __init__(self, arg_names: list[str]):
        self.arg_names = arg_names
        self.statements: list[str] = []
        self.indent = 0

    def _add_stmt(self, text: str):
        self.statements.append("    " * self.indent + text)

    def visit_Module(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node):
        # Only support single target assignment for now
        target = self.visit(node.targets[0])
        value = self.visit(node.value)
        # Check if target is a new local variable
        self._add_stmt(f"CT {target} = {value};")

    def visit_Return(self, node):
        value = self.visit(node.value)
        self._add_stmt(f"((device CT*)out)[elem] = (CT)({value});")

    def visit_For(self, node):
        # Simple for loop: for i in range(N)
        target = self.visit(node.target)
        if (isinstance(node.iter, ast.Call) and 
            isinstance(node.iter.func, ast.Name) and 
            node.iter.func.id == "range"):
            
            args = [self.visit(arg) for arg in node.iter.args]
            if len(args) == 1:
                start, end = "0", args[0]
            else:
                start, end = args[0], args[1]
                
            self._add_stmt(f"for (int {target} = {start}; {target} < {end}; ++{target}) {{")
            self.indent += 1
            for stmt in node.body:
                self.visit(stmt)
            self.indent -= 1
            self._add_stmt("}")
        else:
            raise ValueError("Only simple 'for i in range(...)' loops are supported")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
        }
        op = op_map.get(type(node.op))
        if not op:
            raise ValueError(f"Unsupported binary operator: {type(node.op)}")
        return f"({left} {op} {right})"

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        raise ValueError(f"Unsupported unary operator: {type(node.op)}")

    def visit_Call(self, node):
        func_name = self.visit(node.func)
        # Map mx.xxx to kk_xxx or metal::xxx
        mx_map = {
            "mx.sigmoid": "kk_sigmoid",
            "mx.silu": "kk_silu",
            "mx.tanh": "metal::tanh",
            "mx.exp": "metal::exp",
            "mx.log": "metal::log",
            "mx.abs": "metal::abs",
            "mx.sqrt": "metal::sqrt",
            "mx.rsqrt": "metal::rsqrt",
        }
        if func_name in mx_map:
            func_name = mx_map[func_name]
        elif func_name.startswith("mx."):
            # Fallback for others
            func_name = "kk_" + func_name[3:]
        elif func_name.startswith("metal."):
            func_name = "metal::" + func_name[6:]
        
        args = [self.visit(arg) for arg in node.args]
        return f"{func_name}({', '.join(args)})"

    def visit_Name(self, node):
        return node.id

    def visit_Attribute(self, node):
        value = self.visit(node.value)
        return f"{value}.{node.attr}"

    def visit_Constant(self, node):
        val = node.value
        if isinstance(val, (int, float)):
            return f"(CT){val}"
        return str(val)


def jit(fn: Callable) -> Callable:
    """JIT-compile a Python function to a Metal kernel.
    
    Supported: basic arithmetic, assignments, simple range loops, mx activations.
    Automatically vectorizes memory access using float4 when possible.
    """
    source_code = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source_code)
    
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected a function definition")
        
    arg_names = [arg.arg for arg in func_def.args.args]
    
    visitor = MetalExprVisitor(arg_names)
    visitor.visit(tree)
    
    body = "\n        ".join(visitor.statements)
    
    # Vectorized source using vec<T, 4>
    inputs_src_v = "\n        ".join([f"CT {name} = ((device const CT*) {name}_buf)[elem];" for name in arg_names])
    
    # Scalar source
    inputs_src_s = "\n        ".join([f"CT {name} = {name}_buf[elem];" for name in arg_names])

    source = f"""
        #if VECTORIZED
        #define CT vec<T, 4>
        uint elem = thread_position_in_grid.x;
        {inputs_src_v}
        {body}
        #undef CT
        #else
        #define CT T
        uint elem = thread_position_in_grid.x;
        {inputs_src_s}
        {body}
        #undef CT
        #endif
    """
    
    k = metal_kernel(
        name=f"jit_{fn.__name__}",
        input_names=[f"{name}_buf" for name in arg_names],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

    def wrapper(*args: Any) -> Any:
        if len(args) != len(arg_names):
            raise ValueError(f"Expected {len(arg_names)} arguments, got {len(args)}")
        
        x = args[0]
        size = x.size
        
        # If size is multiple of 4 and aligned, use vectorized path
        if size % 4 == 0:
            return k(
                *args,
                template=[("T", x.dtype), ("VECTORIZED", 1)],
                grid=(size // 4, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[x.shape],
                output_dtypes=[x.dtype],
            )[0]
        
        return k(
            *args,
            template=[("T", x.dtype), ("VECTORIZED", 0)],
            grid=(size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]

    return wrapper
