# Import API class from pexels_api package
from pexels_api import API
# Type your Pexels API
PEXELS_API_KEY = 'z8kHfWcDw785165yssA2sQ4mcKfbW0t7HX6evwV8eMzcyYsjXLdaYyh8'
# Create API object
api = API(PEXELS_API_KEY)
# Search five 'kitten' photos
api.search('cat', page=1, results_per_page=5)
# Get photo entries
photos = api.get_entries()
# Loop the five photos
for photo in photos:
  # Print photographer
  print('Photographer: ', photo.photographer)
  # Print url
  print('Photo url: ', photo.url)
  # Print original size url
  print('Photo original size: ', photo.original)