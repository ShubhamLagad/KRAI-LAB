# 11. Text Mining algorithms on unstructured dataset


from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

text = "In brazil they drive on the right hand side of the road. Brazil has a large coastline on the eastern side of south America"

token = word_tokenize(text)
print(token)

fdist = FreqDist(token)
print(fdist)

# OUTPUT
# ['In', 'brazil', 'they', 'drive', 'on', 'the', 'right', 'hand', 'side', 'of', 'the', 'road', '.', 'Brazil', 'has', 'a', 'large', 'coastline', 'on', 'the', 'eastern', 'side', 'of', 'south', 'America']
# <FreqDist with 20 samples and 25 outcomes>



