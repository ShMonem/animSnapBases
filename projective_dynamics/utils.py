import os

def save_off_mesh(V, F, filename):
    with open(filename, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(V)} {len(F)} 0\n")
        for v in V:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in F:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def check_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)