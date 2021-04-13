import csv
import sys

# ecrit les résultats dans un fichier csv
# Il faut spécifier le chemin du fichier dans path.
def writeIntoFile(over_u, over_v, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):
    path = r"C:\Users\Admin\Desktop\Bachelor Project\GOOOOD\ProjetLTHC\PROJECT\data\data_sans_proj_norm.xls"

    f = open(path, 'a', newline = "")

    w1 = []

    w1.append(N)
    w1.append(M)
    w1.append(lambda_)
    w1.append(beta_u)
    w1.append(beta_v)
    w1.append(lambda_1)
    w1.append(lambda_2)
    w1.append(dt)
    w1.append(over_u)
    w1.append(over_v)

    with f:
        writer = csv.writer(f)
        writer.writerow(w1)