import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision
from mobilenet import MobileNetV2


class DummyNet(nn.Module):
    '''
    dummy network that outputs only zeros
    '''

    def __init__(self, num_classes):
        super(DummyNet, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return torch.zeros(x.shape[0], self.num_classes)



class WeightedAverageNet(nn.Module):
    '''
    simple network calculating a weighted average of the other networks
    '''

    def __init__(self, feature_size, output_size, nb_networks):
        super(WeightedAverageNet, self).__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.nb_networks = nb_networks

        self.fc = []
        for i in range(nb_networks):
            self.fc.append(nn.Linear(in_features = feature_size,
                                     out_features = output_size))

        self.w = torch.ones([nb_networks]) / nb_networks

    def forward(self, x):
        x = x.view(x.shape[0], self.nb_networks, self.feature_size)
        y = torch.empty([x.shape[0], self.nb_networks, self.output_size])
        z = torch.zeros([x.shape[0], self.output_size])

        # bring to proper size
        for i in range(self.nb_networks):
            y[i] = self.fc[i](x[i])

        # weighted average
        for i in range(x.shape[0]):
            for k in range(x.shape[1]):
                z[i] = z[i] + y[i,k]*self.w[k]

        return z



class InputNet(nn.Module):
    '''
    various input networks for the different modalities
    '''

    def __init__(self, input_size, feature_size, model_type):
        super(InputNet, self).__init__()
        self.model_type = model_type

        # CAUTION:
        if model_type == 'resnet18':
            self.model = torchvision.models.resnet18(num_classes = feature_size)
        elif model_type == 'resnet34':
            self.model = torchvision.models.resnet34(num_classes = feature_size)
        elif model_type == 'resnet50':
            self.model = torchvision.models.resnet50(num_classes = feature_size)
        elif model_type == 'resnet101':
            self.model = torchvision.models.resnet101(num_classes = feature_size)
        elif model_type == 'resnet152':
            self.model = torchvision.models.resnet152(num_classes = feature_size)
        elif model_type == 'squeezenet':
            self.model = torchvision.models.squeezenet1_1(num_classes = feature_size)
        elif model_type == 'inceptionnet':
            self.model = torchvision.models.Inception3(num_classes=feature_size, aux_logits=False)
        elif model_type == 'mobilenet':
            self.model = MobileNetV2(input_size = input_size, n_class=feature_size)
        elif model_type == 'dummy':
            self.model = DummyNet(num_classes = feature_size)
        else:
            raise Exception('unknown model type: ' + model_type)

    def forward(self, x):
        return self.model(x)



class AggregatorNet(nn.Module):
    '''
    aggregator network merging the outputs from all input networks
    '''

    def __init__(self, feature_size, output_size, nb_networks, aggregator_type):
        super(AggregatorNet, self).__init__()
        self.aggregator_type = aggregator_type
        self.nb_networks = nb_networks

        if aggregator_type == 'lstm':
            self.model = nn.LSTM(input_size = feature_size*nb_networks,
                                 hidden_size = output_size)
        elif aggregator_type == 'rnn':
            self.model = nn.RNN(input_size = feature_size*nb_networks,
                                hidden_size = output_size)
        elif aggregator_type == 'fc':
            self.model = nn.Linear(in_features = feature_size*nb_networks,
                                   out_features = output_size)
        elif aggregator_type == 'wavg':
            self.model = WeightedAverageNet(feature_size = feature_size,
                                            output_size = output_size,
                                            nb_networks = nb_networks)
        else:
            raise Exception('unknown aggregator type: ' + aggregator_type)

    def forward(self, x):
        if self.aggregator_type == 'lstm' or self.aggregator_type == 'rnn':
            # add extra dimension (weird convention for LSTM and RNN)
            x = x.unsqueeze(1)
            out, hidden = self.model(x)
            # remove extra dimension again
            out = out.squeeze(1)
        else:
            out = self.model(x)
        return out



class CombiNet(nn.Module):
    '''
    network combining the input and aggregator network
    '''

    def __init__(self, input_size, feature_size, output_size, model_types, aggregator_type):
        super(CombiNet, self).__init__()

        # get number of networks
        nb_networks = len(model_types)

        # define all input networks
        self.input_networks = nn.ModuleList()
        for i in range(nb_networks):
            self.input_networks.append(InputNet(input_size = input_size,
                                                feature_size = feature_size,
                                                model_type = model_types[i]))

        # define aggregator network
        self.aggregator_network = AggregatorNet(feature_size = feature_size,
                                                output_size = output_size,
                                                nb_networks = nb_networks,
                                                aggregator_type = aggregator_type)

    def forward(self, x):
        y = None

        # individually calculate the output for the different input networks
        for i in range(len(self.input_networks)):
            # manually upsample data for inception net
            xi = x[i]
            if self.input_networks[i].model_type == 'inception':
                up = nn.Upsample(size=299, mode='bilinear')
                xi = up(xi)

            if y is None:
                y = self.input_networks[i].forward(xi)
            else:
                y = torch.cat((y, self.input_networks[i].forward(xi)), 1)

        # handle combined output of all the input networks in the aggregator network
        y = self.aggregator_network(y)
        return y


