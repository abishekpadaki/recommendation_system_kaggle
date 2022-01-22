import  pandas  as  pd
import numpy as np

class Database:
	"A class representing a database of similaries and common supports"
	def __init__(self, df):
		"the constructor, takes a reviews dataframe like smalldf as its argument"
		database={}
		self.df=df
		'''
		self.uniquebizids={v:k for (k,v) in enumerate(df.ProductId.unique())}
		keys=self.uniquebizids.keys()
		l_keys=len(keys)
		'''
		self.database_sim=np.zeros([155,155])
		self.database_sup=np.zeros([155,155], dtype=np.int)	
	def get_common_tag_no(self,id1,id2):
		"""
		given two ids return the number of common ids
		"""
		data=pd.read_csv("C:/Users/abiks/Desktop/DAProject/KernelTags.csv")
		df1=data[data['KernelId']==id1]
		df2=data[data['KernelId']==id2]
		dfinal = df1.merge(df2, on="TagId", how = 'inner')
		return dfinal.shape[0]
	
	def calculate_similarity(self,i1,i2,df,similarity_func):     #calculates similarity between 2 restaurants using some similarity function
		n_common=self.get_common_tag_no(i1,i2)
		similarity = similarity_func(i1,i2,n_common,self.df)  #using similarity functino defined above to compute similarity
		#checks to see if similarity is NaN and if true, sets similarity to zero
		if np.isnan(similarity): 
			similarity=0

		return (similarity,n_common)              

	def populate_by_calculating(self, similarity_func):

		for i1 in self.df['Id']:
			for i2 in self.df['Id']:
				if i1 < i2:
					sim, nsup=self.calculate_similarity(i1, i2, self.df, similarity_func)
					self.database_sim[i1][i2]=sim
					self.database_sim[i2][i1]=sim
					self.database_sup[i1][i2]=nsup
					self.database_sup[i2][i1]=nsup
				elif i1==i2:   
					sim, nsup=self.calculate_similarity(i1, i2, self.df, similarity_func)
					self.database_sim[i1][i1]=1.
					self.database_sup[i1][i1]=nsup
		print(self.database_sim)

	def get(self, b1, b2):
		"returns a tuple of similarity,common_support given two business ids"	
		sim=self.database_sim[b1][b2]
		nsup=self.database_sup[b1][b2]
		return (sim, nsup)
