from Database import Database
import  pandas  as  pd
import numpy as np
from scipy.stats.stats import pearsonr
import re
import operator
import pickle 
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances


def pearson_sim(i1, i2, n_common, fulldf):
	'''	
	if n_common==0:
        	rho=0.0
	else:
	'''
	rho= pearsonr(fulldf[fulldf['Id']==i1].T,fulldf[fulldf['Id']==i2].T)[0][0]
	'''
	diff1=rest1_reviews['Score']-rest1_reviews['user_avg']
	diff2=rest2_reviews['Score']-rest2_reviews['user_avg']
	rho=pearsonr(diff1, diff2)[0]
	'''
	return rho

def get_common_tag_no(self,id1,id2):
	data=pd.read_csv("C:/Users/abiks/Desktop/DAProject/KernelTags.csv")
	df1=data[data['KernelId']==id1]
	df2=data[data['KernelId']==id2]
	dfinal = df1.merge(df2, on="TagId", how = 'inner')
	'''
	mask = (df.UserId.isin(set_of_users)) & (df.ProductId==restaurant_id)
	reviews = df[mask]
	reviews = reviews[reviews.UserId.duplicated()==False]
	return reviews
	'''
	
	return dfinal.shape[0]
def get_common_dataset_sources(id1,id2):
	data=pd.read_csv("C:/Users/abiks/Desktop/DAProject/KernelVersionDatasetSources.csv")
	df1=data[data['Id']==id1]
	df2=data[data['Id']==id2]
	dfinal = df1.merge(df2, on="SourceDatasetVersionId", how = 'inner')
	return dfinal.shape[0]
	
def calculate_similarity(self,i1,i2,df,similarity_func):     #calculates similarity between 2 restaurants using some similarity function
	n_common=self.get_common_tag_no(i1,i2)
	similarity=similarity_func(i1,i2,n_common,self.df)  #using similarity functino defined above to compute similarity
	#checks to see if similarity is NaN and if true, sets similarity to zero
	if np.isnan(similarity): 
		similarity=0
	return (similarity,n_common)              


def shrunk_sim(sim, n_common, n_common_dataset,reg=3.):
	"takes a similarity and shrinks it down by using the regularizer"
	if(n_common!=0):
		ssim=(n_common*sim)+n_common_dataset/(n_common+reg)
	else:
		ssim=(sim)+n_common_dataset/(n_common+reg)
	return ssim
	
def knearest(id,fulldf,dbase,k,reg):  
	fulldf=fulldf[fulldf.Id!=id]  #takes out the restaurant id provided from the set
	k_list=[]
	for rest_id in fulldf.Id: 
		rest_id=rest_id
		n_common=dbase.get(id, rest_id)[1]        #using .get on instance of database class, gets common users
		sim=dbase.get(id, rest_id)[0]            #using .get on instance of database class, gets similarity
		n_common_dataset=get_common_dataset_sources(id,rest_id)
		shrunk_sim_score=shrunk_sim(sim, n_common, n_common_dataset, reg=reg)    #applies shrunk_sim function to get new similarity after applying reg
		k_list.append((rest_id,shrunk_sim_score,n_common))       #appends the rest id, sim, and common users as a tuple in list
		k_list.sort(key=operator.itemgetter(1),reverse=True)      #sorts the list using shrunk sim
	if k is None:k=7      #if k is not provided, default is set to 7
	return k_list[:k]
	
def biznamefromid(df, theid):
	return df['biz_name'][df['ProductId']==theid].values[0]
def usernamefromid(df, theid):
	return df['ProfileName'][df['UserId']==theid].values[0]

	
def get_user_top_choices(UserId, df, numchoices=5):
	"get the sorted top 5 restaurants for a user by the star rating the user gave them"
	udf=df[df.UserId==UserId][['ProductId','Score']].sort_values(['Score'], ascending=False).head(numchoices)
	return udf
	
def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file. 
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)




fulldf=pd.read_csv("C:/Users/abiks/Desktop/DAProject/KernelsCleaned.csv")
print("read Kernels.csv")

#fulldf=recompute_frame(fulldf)

#computing a subset from large dataframe where number of business reviews are more than 150 and user reviews more than 60
#smalldf=fulldf[(fulldf.product_review_count>150) & (fulldf.user_review_count>60)]  
#smalldf=recompute_frame(smalldf)   #usign the recompute function provided above to re-evaluate the average in smalldf
#smalldf_unique_users=np.unique(smalldf.UserId).size   #getting number of unique users in new df 
#smalldf_items=smalldf.shape[0]     #getting nuber of entries (rows) in new df



fulldf=fulldf.truncate(after=100)



fulldf.drop(fulldf.columns.difference(['Id','TotalViews','TotalComments','AuthorUserId','LanguageName','IsProjectLanguageTemplate','FirstKernelVersionId', 'ForumTopicId', 'TotalVotes']), 1, inplace=True)



'''
db=Database(fulldf)
db.populate_by_calculating(pearson_sim)
save_object(db, 'C:/Users/abiks/Desktop/DAProject/simProducts.pkl')
'''

with open('C:/Users/abiks/Desktop/DAProject/simProducts.pkl', 'rb') as input:
    db= pickle.load(input)

print("got the db")

print("DONE.")


testbizid=125
tops=[]
tops1=knearest(testbizid, fulldf, db, k=7, reg=3.)
print("For ",testbizid, ", top matches are:")
for i, (testbizid, sim, nc) in enumerate(tops1):
	print(i,testbizid, "| Sim", sim, "| Support",nc)
tops.append(tops1)

testbizid=1
tops2=knearest(testbizid, fulldf, db, k=7, reg=3.)
print("For ",testbizid, ", top matches are:")
for i, (testbizid, sim, nc) in enumerate(tops2):
	print(i,testbizid, "| Sim", sim, "| Support",nc)
tops.append(tops2)

testbizid=42
tops3=knearest(testbizid, fulldf, db, k=7, reg=3.)
print("For ",testbizid, ", top matches are:")
for i, (testbizid, sim, nc) in enumerate(tops3):
	print(i,testbizid, "| Sim", sim, "| Support",nc)
tops.append(tops3)

topstotal=tops1+tops2+tops3
topstotal=list(set(topstotal))


personalisation=[]
for i in range(3):
	personalisation_matrix=[]
	for j in range(len(topstotal)):
		if(topstotal[j] in tops[i]):
			personalisation_matrix.append(1)
		else:
			personalisation_matrix.append(0)
	personalisation.append(personalisation_matrix)
for i in range(3):
	for j in range(len(topstotal)):
		print(personalisation[i][j],end=" ")
	print()	

personalisation=np.asarray(personalisation)
print(personalisation)
dist_out = 1-pairwise_distances(personalisation, metric="cosine")
print(dist_out)

print("Personalization Scores: ",1-(sum(( dist_out[i][i+1:] for i in range(len(dist_out)) ), [])/3))
'''
testuserid="A2OEUROGZDTXUJ"
print("For user", usernamefromid(smalldf,testuserid), "top choices are:" )
bizs=get_user_top_choices(testuserid, smalldf)['ProductId'].values
print(bizs)
'''
