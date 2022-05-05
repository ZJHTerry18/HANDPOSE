import numpy as np
class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        ##
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertices.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    ##
                    start = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")-1):
                        end = string.find("/",start)
                        face.append(int(string[start:end]))
                        start = string.find(" ", start) + 1
                    ##
                    self.faces.append(tuple(face))

            f.close()
        except IOError:
            print(".obj file not found.")

def parse_obj(filename):
    O = ObjLoader(filename)
    return np.array(O.vertices),np.array(O.faces).astype(np.int32)-1 # NOTE: need to sub 1!!

if __name__ == "__main__":
    v,f = parse_obj('./finger.obj')
    print(f.shape,f.min())
