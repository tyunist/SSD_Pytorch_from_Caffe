def dict_has_key(dict_instance, key):
   try:
      _has_key = dict_instance.has_key(key)
   except:
      _has_key = key in dict_instance 
   
   return _has_key
