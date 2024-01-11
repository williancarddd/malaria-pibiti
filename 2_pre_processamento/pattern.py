pattern_class = { 
  'gametocyte': 0 ,
  'schizont': 1,
  'trophozoite': 2,
  'leukocyte': 3, 
  'ring': 4, 
  'difficult': 5, 
  'red blood cell': 6
 }

def get_name_image(exame, InstanciaExame, name, ObjectsCategory):
  return f"/images/{exame}-{InstanciaExame}-{name}-{ObjectsCategory}.bmp"