"""Week 5 deep learning project: handwritten digit classification.
Install: pip install torch scikit-learn numpy matplotlib
Run: python week5_pytorch_digit_classification.py
"""
import numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
np.random.seed(42); torch.manual_seed(42)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
digits=load_digits(); X=digits.data.astype(np.float32); y=digits.target.astype(np.int64)
Xa,Xt,ya,yt=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
Xr,Xv,yr,yv=train_test_split(Xa,ya,test_size=.2,random_state=42,stratify=ya)
Xr,Xv,Xt=Xr/16,Xv/16,Xt/16
loader=DataLoader(TensorDataset(torch.tensor(Xr),torch.tensor(yr)),batch_size=32,shuffle=True)
xv,yv=torch.tensor(Xv).to(device),torch.tensor(yv).to(device); xt=torch.tensor(Xt).to(device)
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,10))
    def forward(self,x): return self.layers(x)
model=DigitNet().to(device); loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
best_loss=float("inf"); best_state=None; patience=6; wait=0; best_epoch=0
train_losses=[]; val_losses=[]
for epoch in range(40):
    model.train(); total=0
    for xb,yb in loader:
        xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad()
        loss=loss_fn(model(xb),yb); loss.backward(); optimizer.step(); total+=loss.item()*len(xb)
    model.eval()
    with torch.no_grad(): vloss=loss_fn(model(xv),yv).item()
    train_losses.append(total/len(loader.dataset)); val_losses.append(vloss)
    if vloss<best_loss-1e-4:
        best_loss=vloss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch+1; wait=0
    else: wait+=1
    print(f"Epoch {epoch+1}: train={train_losses[-1]:.4f}, val={vloss:.4f}")
    if wait>=patience: break
model.load_state_dict(best_state); model.eval()
with torch.no_grad(): pred=model(xt).argmax(1).cpu().numpy()
print("Best epoch:",best_epoch,"Accuracy:",accuracy_score(yt,pred))
print(classification_report(yt,pred,digits=4,zero_division=0))
print("Confusion matrix:\\n",confusion_matrix(yt,pred))
plt.plot(train_losses,label="Training"); plt.plot(val_losses,label="Validation"); plt.legend(); plt.savefig("week5_loss.png"); plt.close()
ConfusionMatrixDisplay.from_predictions(yt,pred,cmap="Blues",colorbar=False); plt.savefig("week5_confusion_matrix.png")
torch.save(model.state_dict(),"week5_digit_model.pth")
