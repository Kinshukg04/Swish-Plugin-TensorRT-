import tensorflow as tf
import graphsurgeon as gs



Swish = gs.create_plugin_node(name="Swish_node", op="Swish_TRT")
namespace_plugin_map = { "activation" : Swish }
def preprocess(dynamic_graph):
  dynamic_graph.collapse_namespaces(namespace_plugin_map)
  #dynamic_graph.append("ADD_ANOTHER_CREATE_NODE_PLUGIN_IF_NEEDED")
  #dynamic_graph.remove('input_2') #if you want to remove a node