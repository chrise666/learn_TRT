import onnx
import onnx.helper as helper
import numpy as np

ori_file = r"C:\Users\ADMIN\Documents\WXWork\1688855587383393\Cache\File\2025-02\best.onnx"
onnx_model = onnx.load(ori_file)

graph = onnx_model.graph
print(graph.input[0])
h_in = graph.input[0].type.tensor_type.shape.dim[2]
w_in = graph.input[0].type.tensor_type.shape.dim[3]
h_out = graph.output[0].type.tensor_type.shape.dim[2]
w_out = graph.output[0].type.tensor_type.shape.dim[3]

h_org = h_in.dim_value
w_org = w_in.dim_value
print(h_org, w_org)

h_mod = int(h_org / 2)
w_mod = int(w_org / 2)

h_in.dim_value = h_mod
w_in.dim_value = w_mod
h_out.dim_value = h_mod
w_out.dim_value = w_mod

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, r"C:\Users\ADMIN\Documents\WXWork\1688855587383393\Cache\File\2025-02\best2.onnx")

"""
nodes = [
    helper.make_node(
        name="Conv_0",  # 节点名字，不要和op_type搞混了
        op_type="Conv",  # 节点的算子类型, 比如'Conv'、'Relu'、'Add'这类，详细可以参考onnx给出的算子列表
        inputs=["image", "conv.weight", "conv.bias"],  # 各个输入的名字，结点的输入包含：输入和算子的权重。必有输入X和权重W，偏置B可以作为可选。
        outputs=["3"],
        pads=[1, 1, 1, 1],  # 其他字符串为节点的属性，attributes在官网被明确的给出了，标注了default的属性具备默认值。
        group=1,
        dilations=[1, 1],
        kernel_shape=[3, 3],
        strides=[1, 1]
    ),
    helper.make_node(
        name="ReLU_1",
        op_type="Relu",
        inputs=["3"],
        outputs=["output"]
    )
]
# 关于pads=[1, 1, 1, 1]：
# 第一个值 1：表示在上方添加了一个单位的填充。
# 第二个值 1：表示在下方添加了一个单位的填充。
# 第三个值 1：表示在左侧添加了一个单位的填充。
# 第四个值 1：表示在右侧添加了一个单位的填充。
initializer = [
    helper.make_tensor(
        name="conv.weight",
        data_type=helper.TensorProto.DataType.FLOAT,
        dims=[1, 1, 3, 3],
        vals=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).tobytes(),
        raw=True
    ),
    helper.make_tensor(
        name="conv.bias",
        data_type=helper.TensorProto.DataType.FLOAT,
        dims=[1],
        vals=np.array([0.0], dtype=np.float32).tobytes(),
        raw=True
    )
]

inputs = [
    helper.make_value_info(
        name="image",
        type_proto=helper.make_tensor_type_proto(
            elem_type=helper.TensorProto.DataType.FLOAT,
            shape=["batch", 1, 3, 3]
        )
    )
]

outputs = [
    helper.make_value_info(
        name="output",
        type_proto=helper.make_tensor_type_proto(
            elem_type=helper.TensorProto.DataType.FLOAT,
            shape=["batch", 1, 3, 3]
        )
    )
]

graph = helper.make_graph(
    name="mymodel",
    inputs=inputs,
    outputs=outputs,
    nodes=nodes,
    initializer=initializer
)

# 如果名字不是ai.onnx，netron解析就不是太一样了
opset = [
    helper.make_operatorsetid("ai.onnx", 11)
]

# producer主要是保持和pytorch一致
model = helper.make_model(graph, opset_imports=opset, producer_name="pytorch", producer_version="1.9")
onnx.save_model(model, "edit0.onnx")
print(model)
"""

print("Done.!")