import onnx, onnx_graphsurgeon as gs

NCHW2NHWC = [0,2,3,1]
NHWC2NCHW = [0,3,1,2]

m = onnx.load("model.onnx")
g = gs.import_onnx(m)

# Find the first 4D activation after your Reshape.
# If you know the node name, you can target it directly; otherwise take g.inputs[0]’s first consumer output.
first_out = None
for n in g.nodes:
    if n.op == "Reshape":
        # pick its output tensor as the point to turn NHWC
        first_out = n.outputs[0]
        break
assert first_out is not None, "Couldn't find the initial Reshape; adjust selection."

# Insert NCHW->NHWC
x_nhwc = gs.Variable(first_out.name + "_nhwc", dtype=first_out.dtype)
t_in = gs.Node("Transpose", first_out.name + "_nchw2nhwc", attrs={"perm": NCHW2NHWC},
               inputs=[first_out], outputs=[x_nhwc])
g.nodes.append(t_in)

# Rewire downstream consumers to use NHWC tensor
for n in g.nodes:
    for i,inp in enumerate(n.inputs):
        if inp is first_out:
            n.inputs[i] = x_nhwc

# Insert NHWC->NCHW right before each graph output
for out in g.outputs:
    y = out.inputs[0]
    y_nchw = gs.Variable(out.name + "_nchw", dtype=y.dtype)
    t_out = gs.Node("Transpose", out.name + "_nhwc2nchw", attrs={"perm": NHWC2NCHW},
                    inputs=[y], outputs=[y_nchw])
    g.nodes.append(t_out)
    out.inputs[0] = y_nchw

g.cleanup().toposort()
onnx.save(gs.export_onnx(g), "model_wrap_nhwc.onnx")
