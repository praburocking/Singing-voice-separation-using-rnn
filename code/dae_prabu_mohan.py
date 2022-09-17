import torch

class MyModel(torch.nn.Module):
    def __init__(self,
                 gru_props=[(1025,128,1),(256,128,2),(256,128,1)],
                 lin_props=[(128,1025)]):
        super().__init__()
        self.gru1=torch.nn.GRU(input_size=gru_props[0][0],hidden_size=gru_props[0][1],bidirectional=True,batch_first=True,num_layers=gru_props[0][2])
        self.gru2=torch.nn.GRU(input_size=gru_props[1][0],hidden_size=gru_props[1][1],bidirectional=True,batch_first=True,num_layers=gru_props[1][2])
        self.gru3=torch.nn.GRU(input_size=gru_props[2][0],hidden_size=gru_props[2][1],bidirectional=False,batch_first=True,num_layers=gru_props[2][2])
        self.lin=torch.nn.Linear(in_features=lin_props[0][0],out_features=lin_props[0][1])
        self.activate=torch.nn.ReLU()

    def forward(self,x):

        x=torch.abs(x)
        x,_=self.gru1(x)
        x,_=self.gru2(x)
        x,_=self.gru3(x)
        # print(x.shape)
        x=self.lin(x)

        x=self.activate(x)
        return x
