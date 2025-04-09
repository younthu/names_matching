from UniversityNameMatchBGEV15 import UniversityMatcher
matcher = UniversityMatcher('./data/university_primary_keys.txt')
matches = matcher.find_matches("清华大学")
print(matches)


matcher = UniversityMatcher('./data/university_primary_keys.txt', 'BAAI/bge-m3')
matches = matcher.find_matches("清华大学")
print("=======")
print(matches) 