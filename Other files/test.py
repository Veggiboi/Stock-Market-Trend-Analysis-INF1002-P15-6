from dataclasses import dataclass,replace
@dataclass()
class Living:
    Animals : str
    Fish : str
    Human : int

def chicken():
    
    Animals = "nugget"
    Fish = "tuna"
    Human = 20000
    return Living(Animals,Fish,Human)
x = chicken()
print(x)
x = replace(x, Human = 2)
print(x)
