import numpy as np

class graphneuralnetwork:
    def __init__(self,T,D):
        # Initialize the graph neural network with parameters
        sigma = 0.4
        self.T = T  # Number of aggregation steps
        self.D = D  # Feature vector dimension
        self.W = sigma * np.random.randn(D,D)  # Weight matrix
        self.A = sigma * np.random.randn(D)    # Matrix A
        self.b = 0                             # Bias term

        self.dLdW = np.zeros((D,D))  # Gradient of loss with respect to W
        self.dLdA = np.zeros((D))    # Gradient of loss with respect to A
        self.dLdb = 0                # Gradient of loss with respect to b

    def aggregation1(self,X,adj):
        # Aggregation 1 function: X is the feature vector, adj is the adjacency matrix
        a = np.dot(adj,X)
        return a

    def aggregation2(self,W,a):
        # Aggregation 2 function: W is the weight matrix, a is the result from aggregation 1
        x = np.dot(W,np.transpose(a))
        x = np.transpose(x)
        return x
    
    def relu(self,inp):
        # ReLU activation function
        out = np.maximum(inp,1)
        return out
    
    def readout(self,X):
        # Readout function: Computes the sum of all node's feature vectors
        hG = np.sum(X,axis=0)
        return hG

    def s(self,hG,A,b):
        # Predictor function
        s = np.dot(hG,A)+b
        return s

    def sigmoid(self,s):
        # Sigmoid activation function
        p = 1/(1+np.exp(-s))
        return p

    def output(self,p):
        # Output function: Converts predicted probabilities to binary output
        out = np.where((p>0.5),1,0)
        return out

    def forward(self, nnodes, adj, W = None, A = None, b = None):
        # Forward propagation through the network
        slist = []
        outputlist = []
        X = []
        # feature vector definition
        feat =  np.zeros(self.D)
        feat[0] = 1
        self.tempnnodes = nnodes
        self.tempadj = adj
        if np.any(W == None):
            W = self.W
        if np.any(A == None):
            A = self.A
        if b == None:
            b = self.b
        for i in range(adj.shape[0]):
            X.append(np.tile(feat,[adj.shape[0],1]))
            for j in range(self.T):
                a = self.aggregation1(X[i],adj[i])
                x = self.aggregation2(W,a)
                out = self.relu(x)
                X[i] = out
            hG = self.readout(X[i])
            s = self.s(hG,A,b)
            p = self.sigmoid(s)
            output = self.output(p)
            slist.append(s)
            outputlist.append(int(output))
        return slist,outputlist
    
    def loss(self,s,y):
        # Loss function: Computes loss given predictions and true labels
        losslist = []
        for i in range (len(s)):
            if np.exp(s[i]) > np.finfo(type(np.exp(s[i]))).max:
                loss = y[i]*np.log(1+np.exp(-s[i])) + (1-y[i]) * s[i]  # avoid overflow
            else:
                loss = y[i]*np.log(1+np.exp(-s[i])) + (1-y[i]) * np.log(1+np.exp(s[i]))
            losslist.append(loss)
        return losslist
            
    def update_weight(self,W,A,b):
        # Update network weights and biases
        self.W = W
        self.A = A
        self.b = b

    def backward(self,loss,y,epsilon):
        # Backward propagation to compute gradients
        tempdLdW = np.zeros((self.D,self.D))
        tempdLdA = np.zeros((self.D))
        tempdLdb = 0
        batchsize = len(loss)
        for i in range (self.D):
            for j in range (self.D):
                deltaW = np.zeros((self.D,self.D))
                deltaW[i,j]=epsilon
                Wepsilon = self.W+deltaW
                sep,_ = self.forward(self.tempnnodes,self.tempadj,W=Wepsilon)
                lossep = self.loss(sep,y)
                for k in range(batchsize):
                    tempdLdW[i,j] += (lossep[k] - loss[k])/epsilon
                tempdLdW[i,j] = tempdLdW[i,j]/batchsize
        for i in range (self.D):
            deltaA = np.zeros((self.D))
            deltaA[i] = epsilon
            Aepsilon = self.A + deltaA
            sep,_ = self.forward(self.tempnnodes,self.tempadj,A=Aepsilon)
            lossep = self.loss(sep,y)   
            for j in range(batchsize):
                tempdLdA[i] += (lossep[j] - loss[j])/epsilon
            tempdLdA[i] = tempdLdA[i]/batchsize
        bepsilon = self.b + epsilon
        sep,_ = self.forward(self.tempnnodes,self.tempadj,b=bepsilon)
        lossep = self.loss(sep,y) 
        for i in range(batchsize):
            tempdLdb += (lossep[i] - loss[i])/epsilon
        tempdLdb = tempdLdb/batchsize
        self.dLdW = tempdLdW
        self.dLdA = tempdLdA
        self.dLdb = tempdLdb

# Generating random adjacency matrix and labels
adjacency_matrix = np.random.randint(0, 2, size=(10, 10))
labels = np.random.randint(0, 2, size=(10,))

# Initialize graph neural network
gnn = graphneuralnetwork(T=2, D=10)

# Training loop
epoch = 50
for i in range(epoch):
    predictor_values, _ = gnn.forward(10, adjacency_matrix)
    loss_values = gnn.loss(predictor_values, labels)
    print("Loss Values for epoch:", i+1, "is", np.average(loss_values))
    lr = 0.001
    gnn.backward(loss_values, labels, lr)
    newW = gnn.W - lr * gnn.dLdW
    newA = gnn.A - lr * gnn.dLdA
    newb = gnn.b - lr * gnn.dLdb
    gnn.update_weight(newW, newA, newb)
