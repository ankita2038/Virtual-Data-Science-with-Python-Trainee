import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

SEED=42
OUT="week6_capstone_outputs"
os.makedirs(OUT,exist_ok=True)
wine=load_wine(as_frame=True)
df=wine.frame.copy()
df["class_name"]=df["target"].map({i:n for i,n in enumerate(wine.target_names)})
X=wine.data.copy()
y=wine.target.copy()
df.to_csv(os.path.join(OUT,"wine_dataset.csv"),index=False)
print("Rows:",len(df),"Features:",X.shape[1])
print("Missing:",int(X.isna().sum().sum()),"Duplicates:",int(df.duplicated().sum()))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=SEED,stratify=y)
model=Pipeline([("scale",StandardScaler()),("rf",RandomForestClassifier(n_estimators=300,random_state=SEED,class_weight="balanced"))])
model.fit(X_train,y_train)
pred=model.predict(X_test)
acc=accuracy_score(y_test,pred); prec=precision_score(y_test,pred,average="macro",zero_division=0); rec=recall_score(y_test,pred,average="macro",zero_division=0); f1=f1_score(y_test,pred,average="macro",zero_division=0)
cm=confusion_matrix(y_test,pred)
scaler=StandardScaler(); Z=scaler.fit_transform(X)
km=KMeans(n_clusters=3,random_state=SEED,n_init=20); clusters=km.fit_predict(Z)
sil=silhouette_score(Z,clusters); ari=adjusted_rand_score(y,clusters)
pca=PCA(n_components=2,random_state=SEED); coords=pca.fit_transform(Z); ev=pca.explained_variance_ratio_
df["cluster"]=clusters; df.to_csv(os.path.join(OUT,"wine_dataset_with_clusters.csv"),index=False)
pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"class":y,"cluster":clusters}).to_csv(os.path.join(OUT,"pca_coordinates.csv"),index=False)
X.describe().T.to_csv(os.path.join(OUT,"feature_summary.csv"))
df.groupby("class_name")[wine.feature_names].mean().to_csv(os.path.join(OUT,"class_feature_means.csv"))
pd.Series(clusters).value_counts().sort_index().to_csv(os.path.join(OUT,"cluster_counts.csv"),header=["count"])
imp=pd.Series(model.named_steps["rf"].feature_importances_,index=X.columns).sort_values(ascending=False)
imp.to_csv(os.path.join(OUT,"feature_importance.csv"),header=["importance"])
plt.figure(figsize=(6,4)); df.class_name.value_counts().reindex(wine.target_names).plot(kind="bar"); plt.title("Wine samples by class"); plt.xlabel("Class"); plt.ylabel("Samples"); plt.tight_layout(); plt.savefig(os.path.join(OUT,"class_distribution.png"),dpi=160); plt.close()
corr=X.corr(); plt.figure(figsize=(9,7)); plt.imshow(corr,vmin=-1,vmax=1,aspect="auto"); plt.colorbar(label="Correlation"); plt.xticks(range(len(corr)),corr.columns,rotation=90,fontsize=7); plt.yticks(range(len(corr)),corr.columns,fontsize=7); plt.title("Feature correlation matrix"); plt.tight_layout(); plt.savefig(os.path.join(OUT,"correlation_matrix.png"),dpi=160); plt.close()
plt.figure(figsize=(6,4)); plt.scatter(coords[:,0],coords[:,1],c=y); plt.title("PCA projection by known class"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.savefig(os.path.join(OUT,"pca_by_class.png"),dpi=160); plt.close()
plt.figure(figsize=(6,4)); plt.scatter(coords[:,0],coords[:,1],c=clusters); plt.title("K-Means clusters in PCA space"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.savefig(os.path.join(OUT,"pca_by_cluster.png"),dpi=160); plt.close()
plt.figure(figsize=(5,4)); plt.imshow(cm); plt.colorbar(); plt.xticks(range(3),wine.target_names,rotation=20); plt.yticks(range(3),wine.target_names)
for i in range(3):
 for j in range(3): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Random Forest confusion matrix"); plt.tight_layout(); plt.savefig(os.path.join(OUT,"confusion_matrix.png"),dpi=160); plt.close()
plt.figure(figsize=(7,4)); imp.head(10).sort_values().plot(kind="barh"); plt.title("Top 10 feature importances"); plt.xlabel("Importance"); plt.tight_layout(); plt.savefig(os.path.join(OUT,"feature_importance.png"),dpi=160); plt.close()
print("Accuracy:",round(acc,4),"Macro precision:",round(prec,4),"Macro recall:",round(rec,4),"Macro F1:",round(f1,4))
print("Silhouette:",round(sil,4),"ARI:",round(ari,4),"PCA variance:",np.round(ev,4).tolist(),"sum:",round(float(ev.sum()),4))
print("Class report:\\n",classification_report(y_test,pred,target_names=wine.target_names,zero_division=0))
print("Clusters:",pd.Series(clusters).value_counts().sort_index().to_dict())
print("Top features:",imp.head(6).round(4).to_dict())
