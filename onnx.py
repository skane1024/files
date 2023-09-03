import onnx

# 加载ONNX模型
model = onnx.load('model/onnx/mnist-12.onnx')




def get_tensor_shape(model, name):
    tensor_shape = None
    for tensor in model.graph.initializer:
        if tensor.name == name:
            input_tensor = tensor
            tensor_shape = [dim.dim_value for dim in input_tensor.dims]
    
    for input in model.graph.input:
        if input.name == name:
            tensor_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
            break

    for output in model.graph.output:
        if output.name == name:
            tensor_shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            break
    
    for vf in model.graph.value_info:
        if vf.name == name:
            tensor_shape = [dim.dim_value for dim in vf.type.tensor_type.shape.dim]
    return tensor_shape
    


def conv_parameters(model):
    for node in model.graph.node:
        if node.op_type == 'Conv':
            input_name = node.input[0]
            output_name = node.output[0]
            input_shape= get_tensor_shape(model, input_name)
            output_shape = get_tensor_shape(model, output_name)
            
            kernel_size = None
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    kernel_size = attr.ints
                    break
                
            strides = None
            for attr in node.attribute:
                if attr.name == 'strides':
                    strides = attr.ints
                    break
                
            padding = None
            for attr in node.attribute:
                if attr.name == 'padding':
                    padding = attr.ints
                    break
                
            dilations = None
            for attr in node.attribute:
                if attr.name == 'dilations':
                    dilations = attr.ints
                    break
                
            group = None
            for attr in node.attribute:
                if attr.name == 'group':
                    group = attr.ints
                    break
            
            # # 打印参数信息
            print('Conv Parameters:')
            print('Input shape:', input_shape)
            print('Output shape:', output_shape)
            print('Kernel Size:', kernel_size)
            print('Stride:', strides)
            print('Padding:', padding)
            print('group:', group)
            print('---------------------')
            
            # print(node.attribute)
            

def maxpool_parameters(model):
    for node in model.graph.node:
        if node.op_type == 'MaxPool':
            input_name = node.input[0]
            output_name = node.output[0]
            input_shape= get_tensor_shape(model, input_name)
            output_shape = get_tensor_shape(model, output_name)
                    
            kernel_size = None
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    kernel_size = attr.ints
                    break
                
            strides = None
            for attr in node.attribute:
                if attr.name == 'strides':
                    strides = attr.ints
                    break
                
            pads = None
            for attr in node.attribute:
                if attr.name == 'pads':
                    pads = attr.ints
                    break
            print('maxpool Parameters:')
            print('Input shape:', input_shape)
            print('Output shape:', output_shape)
            print('Kernel Size:', kernel_size)
            print('Stride:', strides)
            print('pads:', pads)
            print('---------------------')
            
            

def relu_parameters(model):
    for node in model.graph.node:
        if node.op_type == 'Relu':
            input_name = node.input[0]
            input_shape= get_tensor_shape(model, input_name)
            print('Relu Parameters:')
            print('Input shape:', input_shape)
            print('---------------------')
            # print(node.attribute)



def reshape_parameters(model):
    for node in model.graph.node:
        if node.op_type == 'Reshape':
            input_name = node.input[0]
            output_name = node.output[0]
            input_shape= get_tensor_shape(model, input_name)
            output_shape = get_tensor_shape(model, output_name)
            # # 打印参数信息
            print('reshape Parameters:')
            print('Input shape:', input_shape)
            print('Output shape:', output_shape)
            print('---------------------')
            
            # print(node.attribute)

conv_parameters(model)
