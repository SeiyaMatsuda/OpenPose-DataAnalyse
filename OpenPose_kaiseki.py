#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from cmath import phase
from math import degrees
#最大人数分のリストを作成する関数make_list
def make_list(num):
    kansetsu_data=[[]]*num
    return kansetsu_data
# jsonファイルの格納ディレクトリ取得
file_path=input("ディレクトリのパスを入力").strip("\"")+"\*"
json_file=glob.glob(file_path)
#身体の６つの角の組み合わせの定義
pairs=[(1,2,3),(1,5,6),(2,3,4),(5,6,7),(8,9,10),(11,12,13)]
#動画像内の最大人数の確認
max_num=0
for j in json_file:
    with open(j,'r') as f:
        data = json.load(f)
    if max_num<len(data['people']):
        max_num=len(data['people'])
#角度変化を格納するリスト(angle_data)の定義
angle_data=make_list(max_num)
#データ整列に使うx座標を取得する処理
#1.首のx座標を格納するリスト(x_list)を定義
x_list=make_list(len(json_file))
#2.jsonファイルを読み込み首のX座標を格納
for c,j in enumerate(json_file):
    with open(j,'r') as f:
        data = json.load(f)
        for d in data['people']: 
                #jsonファイルを整列
            keypoints= np.array(d['pose_keypoints_2d']).reshape((25, 3))
                #x座標を取りだし格納
            x_list[c]=x_list[c]+[keypoints[1][0]]
#3.検出されなかったx座標の処理
for r1 in range(1,len(x_list)):
    if 0.0 in x_list[r1]:
        x_notin0=[i for i in x_list[r1] if i!=0]
        del_index=[]
        for r2 in range(0,len(x_notin0)):
            del_x=[]
            for r3 in range(0,len(x_list[r1-1])):
                del_x+=[abs(x_notin0[r2]-x_list[r1-1][r3])]
                del_index+=[del_x.index(min(del_x))]
        q1=list(range(0,len(x_list[r1-1])))
        q2=list(set(q1)-set(del_index))
        x_list[r1][x_list[r1].index(0.0)]=x_list[r1-1][q2[0]]        
    else:
        continue
#順番を定義(左の順番から並び替える)
A=np.array(x_list[0])
A=np.argsort(A)
index=A.reshape(len(A),1).tolist()
#それぞれの人物のインデックスを取得(index=[[人物A],[人物B]...])
for i,p in enumerate(A):
    #1.インデックスを取得する人物
    n=p
    #2.1フレーム目から最終フレーム目まで処理を行う
    for r in range(len(x_list)-1):
        #3.nフレーム目とn+1フレーム目での誤差を格納するx_difを定義
        x_dif=[]
        for c in range(len(x_list[r+1])):
                x_dif+=[abs(x_list[r+1][c]-x_list[r][n])]
        #4.誤差の最小値より各人物のインデックスを取得
        index[i]=index[i]+[x_dif.index(min(x_dif))]
        n=x_dif.index(min(x_dif))
#処理するjsonファイルのナンバーを定義(1～最終フレームまで)
frame_number=1        
#jsonファイルに対するループ処理
for j in json_file:
    #angle_dataにフレーム番号を追加
    for r in range(0,len(angle_data)):
        angle_data[r]=angle_data[r]+[frame_number]
    #jsonファイルの読み込み
    with open(j,'r') as f:
        data = json.load(f)
        #処理する人数の定義
        person_number=0
        #格納リストの更新
        #人数が最大人数以下の場合、検出されなかった人物の角度データをNoneで補う
        for a in range(len(data['people']),len(angle_data)):
            angle_data[a]=angle_data[a]+[None,None,None,None,None,None]     
        #データの整列
        index_number=[]
        #index_numberに順番を格納して、jsonを読み込んだデータの人物の順番を書き換える
        for i in range(0,len(index)):
              index_number=index_number+[index[i][frame_number-1]]
        data["people"]=[data["people"][index_number[i]] for i in range(len(index_number))]
        #*検出された全員について
        for d in data['people']: 
            #1.jsonファイルをNumpyを用いて整列
            kpt = np.array(d['pose_keypoints_2d']).reshape((25, 3))
            #2.組み合わせに対し角度を算出
            for p in pairs:
                #1点目
                pt1 = list(list(map(int, kpt[p[0], 0:2])))
                #1点目の信頼値
                c1 = kpt[p[0], 2]
                #2点目
                pt2 = list(list(map(int, kpt[p[1], 0:2])))
                #2点目の信頼値
                c2 = kpt[p[1], 2]
                #3点目
                pt3 = list(list(map(int, kpt[p[2], 0:2])))
                #3点目の信頼値
                c3 = kpt[p[2], 2]
                   
                #信頼度0.0のキーポイントは無視（無視した場合、角度は算出できないためNoneを追加）
                if c1 == 0.0 or c2 == 0.0 or c3 == 0.0 :
                    angle_data[person_number]+=[None]            
                #信頼性を持つ結果に対して角度を算出    
                else:
                    #1.それぞれのキーポイント情報を複素数平面に変換
                    a=complex(pt1[0],pt1[1])
                    b=complex(pt2[0],pt2[1])
                    c=complex(pt3[0],pt3[1])
                    ba=a-b
                    bc=c-b
                    #2.角度算出不可の情報を消去(二つのキーポイントが一致する場合算出不可)
                    if bc==0:
                        angle_data[person_number]+=[None] 
                    #3.(1)でない場合,角度を計算しデータをリスト化                
                    else:
                        angle=degrees(phase(ba/bc))
                        #4.算出した角度を0～360°に変換
                        if angle<0:
                            angle=angle+360
                        #5.算出した角度データを追加
                        angle_data[person_number]+=[angle] 
           #処理する人物の更新                
            person_number = person_number+1
    #処理するフレームの更新
    frame_number+=1
#最終的な角度データをNumpyを用いて整列したresult_angleを得る
result_angle=[]
for c in range(0,len(angle_data)):
    result_angle+=[np.array(angle_data[c]).reshape(int(len(angle_data[c])/7),7)]


# In[2]:


#外れ値の変換を行う
#*すべての人物に対しループ処理
for r in range(0,len(result_angle)):
    #*すべてのフレームに対しループ処理
    for i in range(0,len(result_angle[r])-1):
        #*身体の6つの角についてループ処理
        for s in range(1,len(result_angle[r][i])):
            #角度の算出ができてない場合、前の値を参照
            if result_angle[r][i+1][s]==None:
                    result_angle[r][i+1][s]=result_angle[r][i][s]
            else:
                #前フレームと比べて誤差が50以上ならば、前フレームを算出
                if abs(result_angle[r][i+1][s]-result_angle[r][i][s])>50:
                    result_angle[r][i+1][s]=result_angle[r][i][s]
                else:
                    continue


# In[3]:


#評価尺度
#手法(平均値との誤差を求める関数)dif(人物のインデックス)
def dif(i):
    dif_angle=[[]]*6
    #*すべての身体の６つの角に対してループ処理
    for a in range(1,7):
        #1.角度平均を算出
        #avarage_listを初期化
        avarage_list=[]
        #*第一フレームから最終フレームまでループ
        for r in range(0,len(result_angle[i])):
            #avarage_listに最終フレームまでのデータ値を格納
            if result_angle[i][r][a]==None:
                continue
            avarage_list+=[result_angle[i][r][a]]
        #avarage_listにデータが格納されてない場合
        if len(avarage_list)==0:
            avarage=0.0
        #average_listより角度平均avarageを算出
        else:    
            avarage=sum(avarage_list)/len(avarage_list)
        #2.dif_angleにavarageと算出した角度データの誤差を格納
        for r in range(0,len(result_angle[i])):
            if result_angle[i][r][a]==None:
                dif_angle[a-1]=dif_angle[a-1]+[None]
            else:
                dif=abs(result_angle[i][r][a]-avarage)              
                dif_angle[a-1]=dif_angle[a-1]+[dif]
    #3.dif_angleを結果として返す
    return dif_angle
#dif_angleに一人分の角度の誤差は格納[組み合わせごとの角度,フレーム]
#新たにdif_angle_avarageに平均値との誤差の組み合わせの平均値を全員について格納[人,組み合わせごとの角度]
dif_angle_average=[[]]*len(result_angle)
#*人数分ループ
for r in range(0,len(result_angle)):
    #平均値との角度誤差を取り出す
    dif_angle=dif(r)
    for p in range(0,len(dif_angle)):
        dif_angle[p]=[i for i in dif_angle[p] if i!=None]
        dif_angle_average[r]=dif_angle_average[r]+[sum(dif_angle[p])/len(dif_angle[p])]


# In[23]:


#データファイルを得点にしたリストを返す関数tokuten(人)
import math
import string
#i得点を算出する関数
def tokuten(a):
    score=[[]]*len(a)
    #*リストの長さ（人数分）のループ処理
    for p in range(0,len(a)):
        #*身体の６つの角についてのループ処理
        for r in range(0,len(a[p])):
            #データが0°の時,得点は０
            if a[p][r]==0.0:
                tokuten=0.0
            #データが実数値のとき、対数関数より得点を算出
            else:
                s1=math.log(a[p][r]+1)
                s2=math.log(360+1)
                tokuten=10*(s1/s2)
            score[p]=score[p]+[tokuten]
    #得点を10点満点で返す
    return score
#点数化
score=tokuten(dif_angle_average)
#アルファベットのリストを作成
alph=list(string.ascii_uppercase)
#しきい値を算出
#1.ユーザーにより動画像の撮影条件を入力
distance=float(input("対象からの距離を入力"))
height=float(input("対象からの高さを入力"))
kiso_score=float(input("基礎の得点を入力"))
#2.距離が0の時、cosθ=0
if distance==0:
    w=0
#3.距離により,cosθを求める
else:
    w=math.sqrt(1/(1+(height/distance)**2))
#4.しきい値Kの補正
K=kiso_score*w
#6角に対し重み付けを行う
w0,w1,w2,w3,w4,w5=(float(w) for w in input("重みw0～w5を入力（0.0<w≦1.0）*数値間に空白を開けてください").split())
#点数を合計し得点率に換算
for s in range(0,len(score)):
    #定義した重みと得点を掛け合わせる
    score[s]=[score[s][0]*w0,score[s][1]*w1,score[s][2]*w2,score[s][3]*w3,score[s][4]*w4,score[s][5]*w5]
    #得点をパーセンテージに変換
    parcent=((sum(score[s])/(10*(w0+w1+w2+w3+w4+w5)))*100)
#得点率より状態を判定
    #得点を出力
    print(parcent)
    if parcent>=K:
        print("人物"+alph[s]+"は動作しています")
    else:
        print("人物"+alph[s]+"は静止しています")


# In[7]:


#時系列データ作成関数make_graf(人,pair)
def make_graf(h,p):
    x=[]
    y=[]
    for r in range(0,len(result_angle[0])):
        x=x+[result_angle[h][r][0]]
        if result_angle[h][r][p]==None:
            if len(y)==0:
                y=y+[None]
            else: 
                y=y+[y[-1]]
        else:
            y=y+[result_angle[h][r][p]]
    return x,y
x,y1=make_graf(1,1)
x,y2=make_graf(1,2)
x,y3=make_graf(1,3)
x,y4=make_graf(1,4)
x,y5=make_graf(1,5)
x,y6=make_graf(1,6)
for y in y6:
    if  300<y:
        y6[y6.index(y)]=y-360
plt.plot(x,y1,label="i=1")
plt.plot(x,y2,label="i=2")
plt.plot(x,y3,label="i=3")
plt.plot(x,y4,label="i=4")
plt.plot(x,y5,label="i=5")
plt.plot(x,y6,label="i=6")
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.tight_layout()
plt.xlabel("フレーム数",fontname="MS Gothic")
plt.ylabel("角度",fontname="MS Gothic")
plt.show()


# In[8]:


make_graf(0,2)
make_graf(1,2)
make_graf(2,2)
make_graf(3,2)


# In[9]:


x_graf=make_list(max_num)
for p in range(0,len(index)):
    for i,j in enumerate(index[p]):
        x_graf[p]=x_graf[p]+[x_list[i][j]]
x=list(range(0,len(x_graf[0])))
y1=x_graf[0]
y2=x_graf[1]
y3=x_graf[2]
y4=x_graf[3]


# In[10]:


x_graf=make_list(max_num)
for p in range(0,len(x_list[i])):
    for i in range(0,len(x_list)):
         x_graf[p]=x_graf[p]+[x_list[i][p]]
x=list(range(0,len(x_graf[0])))
print(len(x_graf[0]))

y1=x_graf[0]
y2=x_graf[1]
y3=x_graf[2]
y4=x_graf[3]


# In[11]:


plt.plot(x,y1,label="A")
plt.plot(x,y2,label="B")
plt.plot(x,y3,label="C")
plt.plot(x,y4,label="D")
plt.legend()
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.xlabel("フレーム数",fontname="MS Gothic")
plt.ylabel("首のX座標",fontname="MS Gothic")


# In[12]:


for p in range(0,len(y1)):
    print([y1[p],y2[p],y3[p],y4[p]])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




