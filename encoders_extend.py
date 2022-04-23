import time

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

from set2set import Set2Set

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            # 根据p随机将输入张量元素设置为0
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        #将权重矩阵注册
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        #self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
            #self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        #add_self -> False
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            # y->[20, 1000, 30]
            # 对第3个维度进行L2归一化
            y = F.normalize(y, p=2, dim=2)
            #print(y[0][0])
        return y

class GcnEncoderGraph(nn.Module):
    # input_dim->特征维度；embedding_dim->output_dim；label_dim->num_classes；num_layers->num_gc_layers
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        # add_self->False
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1

        self.bias = True
        # 默认为True
        if args is not None:
            self.bias = args.bias
        # 3个卷积层
        '''
            conv_first->[3, 30]
            conv_block->[30, 30]
            conv_last->[30, 30]
        '''
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim
        # ??
        # hidden_dim=30; num_layers=3; embedding_dim=30

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        #预测模型
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)
        # 返回该网络中的所有modules，继承nn.Module的模型
        # 为所有继承nn.Module模型的权重初始化
        for m in self.modules():
            # isinstance->判断一个对象是否是某一类型
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last
    def build_VGAE_layers(self, input_dim, hidden_dim, output_dim, normalize=False):
        '''
             创建VGAE
        '''
        base_gcn = GraphConv(input_dim = input_dim, output_dim = hidden_dim, add_self = False,
                             normalize_embedding=normalize, bias=self.bias)
        gcn_mean = GraphConv(input_dim=hidden_dim, output_dim=output_dim, add_self=False,
                             normalize_embedding=normalize, bias=self.bias)
        gcn_out = GraphConv(input_dim=hidden_dim, output_dim=output_dim, add_self=False,
                             normalize_embedding=normalize, bias=self.bias)
        return base_gcn, gcn_mean, gcn_out
    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes): 
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        # batch_num_nodes中储存了当前batch里面每一个图的节点数量
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()
        #在最后增加一维[batch_size x max_nodes x 1]
        #return out_tensor.unsqueeze(2)

    def construct_mask_extend(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        # batch_num_nodes中储存了当前batch里面每一个图的节点数量
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i], :batch_num_nodes[i]] = mask
        return out_tensor
        #在最后增加一维[batch_size x max_nodes x 1]
        #return out_tensor.unsqueeze(2)

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        #对第二维进行batch归一化
        #bn_module = nn.BatchNorm1d(x.size()[1])
        return bn_module(x)
    '''
        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        conv_first->[3, 30]
        conv_block->[30, 30]
        conv_last->[30, 30]
        assign_conv_first[3, 30]
        assign_conv_block[30, 30]
        assign_conv_last[30, 100]
    '''

    def vgae_forward(self, x, adj, cluster, batch_num_nodes, adj_origin, base_gcn, gcn_mean, gcn_logstd, embedding_mask=None):
        '''
            perform forward prop with VGAE
        Returns:
            Reconstruct matrix with dimension [batch_szie x num_nodes x num_nodes]
        '''
        hidden = base_gcn(x, adj)
        self.mean = gcn_mean(hidden, adj)
        self.logstd = gcn_logstd(hidden, adj)
        # self.bn->True
        # batch Normalization
        if self.bn:
            x = self.apply_bn(x)
        laplace = torch.distributions.laplace.Laplace(torch.tensor(0.0), torch.tensor(1.0))
        laplace_dis = laplace.sample(self.mean.shape).cuda()
        sample_z = laplace_dis * torch.exp(self.logstd) + self.mean
        #print('sample_z:',sample_z.shape)
        #print('assign:',cluster.shape)
        extends = cluster @ sample_z
        #print('extends:',extends.shape)
        #exit(0)
        #print(sample_z.shape)
        #print(torch.count_nonzero(cluster[0]).item())
        #print(torch.count_nonzero(cluster[1]).item())
        #print(torch.count_nonzero(cluster[2]).item())
        #print(adj.shape)
        #extends = torch.zeros((cluster.shape[0], cluster.shape[1], sample_z.shape[2]))
        #print(cluster)
        # j->batch_id; k->batch_num
        # 每一个节点，根据其所分配到的簇，提取特征
        '''for j, k in zip(range(len(cluster)), batch_num_nodes):
            # 对于batch中的每一个节点
            for m in range(k):
                # 当前节点m被映射到簇map_num中
                map_num = cluster[j][m]
                # batch——j 拓展中的第m个节点的特征，来自于batch——j的采样的第map_num 个簇
                extends[j][m] = sample_z[j][map_num]'''
        # extends: [20, 1000, 30]
        #print(extends.shape)
        '''for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        # 按照第三个维度进行拼接 [20, 1000, 90] ; assign: [20, 1000, 160]
        x_tensor = torch.cat(x_all, dim=2)
        # embedding_mask的维度为[20, 1000, 1] 与 x_tensor维度不同，在进行相乘时，会首先将embedding_mask的维度拓展为[20, 1000, 90]
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask'''
        # 返回经过重采样且经过id映射维度拓展的embedding
        return extends

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''
        x = conv_first(x, adj)
        x = self.act(x)
        # self.bn->True
        # batch Normalization
        if self.bn:
            x = self.apply_bn(x)
        # 加残差？(graph_sage)
        x_all = [x]
        #out_all = []
        #out, _ = torch.max(x, dim=1)
        #out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x,adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        # 按照第三个维度进行拼接 [20, 1000, 90] ; assign: [20, 1000, 160]
        x_tensor = torch.cat(x_all, dim=2)
        # embedding_mask的维度为[20, 1000, 1] 与 x_tensor维度不同，在进行相乘时，会首先将embedding_mask的维度拓展为[20, 1000, 90]
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out,_ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x,adj)
        #x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        #print(output.size())
        return ypred

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            # F.cross_entropy输入pred为[N, C]，C为类别个数，首先对pred使用softmax，然后对label使用onehot计算损失
            # reduction='mean' 表示损失取平均
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1,1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)
            
        #return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class GcnSet2SetEncoder(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnSet2SetEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims, concat, bn, dropout, args=args)
        self.s2s = Set2Set(self.pred_input_dim, self.pred_input_dim * 2)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out = self.s2s(embedding_tensor)
        #out, _ = torch.max(embedding_tensor, dim=1)
        ypred = self.pred_model(out)
        return ypred

'''
    model = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim).cuda()
'''
class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
            assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''
        #子类SoftPoolingGcnEncoder显式调用父类GcnEncoderGraph的初始化方法
        #input_dim->特征维度；embedding_dim->output_dim；label_dim->num_classes；num_layers->num_gc_layers
        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        # GC
        #注册3个卷积层，参数被注册
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        self.vgae_first_after_pool = nn.ModuleList()
        self.vgae_block_after_pool = nn.ModuleList()
        self.vgae_last_after_pool = nn.ModuleList()
        '''
            self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
                
                conv_first2:[90, 30]
                conv_block2:[30, 30]
                conv_last2:[30, 30]
        '''
        # 将conv_after_pool 换为 VGAE
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers, 
                    add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)
            VGAE_first2, VGAE_block2, VGAE_last2 = self.build_VGAE_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, normalize=True)
            self.vgae_first_after_pool.append(VGAE_first2)
            self.vgae_block_after_pool.append(VGAE_block2)
            self.vgae_last_after_pool.append(VGAE_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            #num_gc_layers
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            #feature_dim
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        # 使用max_nodes和assign_ratio规定了池化后的节点数量
        assign_dim = int(max_num_nodes * assign_ratio)
        self.assign_dim = assign_dim
        #print('assign:', assign_dim)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            # assign_hidden_dim初始化为hidden_dim
            '''
                assign_conv_first[3, 30]
                assign_conv_block[30, 30]
                assign_conv_last[30, 100]
            '''
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            # assign_pred: [160, 100]
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)
            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)
        # pred_model:[180, 50, 6]
        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling+1), pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)
        self.decoder_nn = self.build_pred_layers(embedding_dim, [max_num_nodes // 10, max_num_nodes // 2], max_num_nodes, num_aggs=1)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)
    '''
        model(h0, adj, batch_num_nodes, assign_x=assign_input)
    '''
    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x
        adj_origin = adj
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None
        #print(batch_num_nodes)
        out_all = []
        # 使用gcn_forward求出嵌入
        # 使用了3层卷积网络，并加入残差
        '''
            conv_first->[3, 30]
            conv_block->[30, 30]
            conv_last->[30, 30]
        '''
        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        embedding = embedding_tensor.clone()
        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        # self.num_aggs = 1
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None
            max_batch_num_nodes = max(batch_num_nodes)
            #print(batch_num_nodes)
            # x_a 与 x 相同
            '''
                assign_conv_first[3, 30]
                assign_conv_block[30, 30]
                assign_conv_last[30, 100]
            '''
            self.assign_tensor = self.gcn_forward(x_a, adj, 
                    self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i],
                    embedding_mask)

            # [batch_size x num_nodes x next_lvl_num_nodes]
            # 对输入数据的最后一维元素求softmax
            # assign_pred: [160, 100]
            # 经过3层assign卷积，然后经过线性预测层，最后经过softmax输出
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask
            #exit(0)
            # update pooled features and adj matrix
            # 将assign_tensor的 1，2维度互换（转置）
            #  X = S^T Z
            # assign_tensor:[20, 1000, 100]; embedding_tensor:[20, 1000, 90]
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            # A = S^T A S; @ -> tensor中的矩阵乘法
            # adj:[20, 1000, 1000];
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x
            # 每个节点被分配到的簇的下标 [20, 1000]
            #print(self.assign_tensor)
            argmax_list = torch.argmax(self.assign_tensor, dim=2)
            #cluster_list = torch.zeros((argmax_list.shape[0], self.assign_dim))
            #print(argmax_list)
            '''for j,k in zip(range(len(argmax_list)),batch_num_nodes):
                for m in range(k):
                    cluster_list[j][argmax_list[j][m]] += 1'''
            # cluster_list 保存了每一个簇中的节点数量
            # 池化后卷积
            '''
                conv_first2:[90, 30]
                conv_block2:[30, 30]
                conv_last2:[30, 30]
            '''
            # x:[20, 100, 90]; adj:[20, 100, 100]
            embedding_tensor = self.gcn_forward(x, adj,
                    self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                    self.conv_last_after_pool[i])
            positive = embedding_tensor.clone()
            extend_embedding_tensor = self.vgae_forward(x, adj, self.assign_tensor, batch_num_nodes, adj_origin,
                                                self.vgae_first_after_pool[i], self.vgae_block_after_pool[i],
                                                self.vgae_last_after_pool[i])
            #print(time1-time0)
            # embedding_tensor:[20, 100, 90]
            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                #out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            # output:[20, 90] + [20, 90] = [20, 180]
            output = torch.cat(out_all, dim=1)

        else:
            output = out
        # pred_model:[180, 50, 6]; ypred:[20, 6]
        ypred = self.pred_model(output)
        return ypred, extend_embedding_tensor, embedding

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        # self.linkpred -> False
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            # self.assign_tensor:[20, 1000, 100]; pred_adj0:[20, 1000, 1000]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop-1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            # pred_adj中的每一个元素，大于1的取1，小于1的不变
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
            #pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype))
            #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1-adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            #print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss

    def extend_loss(self, pred, adj, batch_num_nodes, weight_tensor = None):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        #loss = super(SoftPoolingGcnEncoder, self).loss(pred, adj, weight_tensor)
        #pred = self.dot_product_decode(pred).cuda()
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask_extend(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None
        pred = self.decoder_nn(pred.cuda())
        #pred = self.dot_product_decode(pred).cuda()
        if embedding_mask is not None:
            pred = pred * embedding_mask.cuda()
        pred = torch.sigmoid(pred)
        #print(pred.shape)
        #print(adj.shape)
        F_loss = F.binary_cross_entropy(pred.view(-1), adj.view(-1))
        #self.link_loss = -adj * torch.log(pred + eps) - (1 - adj) * torch.log(1 - pred + eps)
        #num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
        #print(torch.sum(self.link_loss) / float(num_entries))
        # self.linkpred -> False
        return F_loss

    def dot_product_decode(self, Z):
        '''
            乘积解码
        '''
        A_pred = Z @ torch.transpose(Z, 1, 2)
        return A_pred

    def MI_Est(self, discriminator, embeddings, positive):
        '''

        :param discriminator: 辨别器：[3*out_put,1]
        :param embeddings: 图级别嵌入：[20, 1000, 90]
        :param positive: 子图复原嵌入：[20, 1000, 30]
        :return:
        '''
        # torch.randperm(self.batch_size) -> 将(0, self.batch_size)随机打乱获得一个数字序列
        shuffle_embeddings = embeddings[torch.randperm(embeddings.shape[0])]
        # joint:[128, 1]
        joint = discriminator(embeddings,positive)
        # margin:[128, 1]
        margin = discriminator(shuffle_embeddings,positive)
        mi_est = torch.sigmoid(torch.mean(joint) - torch.log(torch.mean(torch.exp(margin))))

        return mi_est

class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.args = args
        # input_size: output_dim * 4
        self.input_size = self.args.output_dim * 4
        # hidden_size: output_dim
        self.hidden_size = self.args.output_dim
        # fc1:[input_size, hidden_size]
        self.fc1 = nn.Linear(self.input_size,self.hidden_size).cuda()
        # fc2:[hidden_size, 1]
        self.fc2 = nn.Linear(self.hidden_size, 1).cuda()
        self.relu = nn.ReLU()

        torch.nn.init.constant_(self.fc1.weight, 0.01)
        torch.nn.init.constant_(self.fc2.weight, 0.01)

    def forward(self, embeddings,positive):

        # 按照倒数第一维进行拼接 [128, 8] cat(-1) [128, 8] -> [128, 16]
        cat_embeddings = torch.cat((embeddings, positive),dim = -1)
        #print(cat_embeddings.shape)
        # pre:[128, 4]
        pre = self.relu(self.fc1(cat_embeddings))
        # pre:[128, 1]
        pre = self.fc2(pre)

        return pre