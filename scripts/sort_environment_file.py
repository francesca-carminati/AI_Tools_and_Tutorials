import fire
import yaml


def main(in_path="environment.yml", out_path="environment.yml"):
    # Open the file and load the file
    with open(in_path, "r") as f:
        environment_file = yaml.load(f, Loader=yaml.FullLoader)

    # Sort dependencies
    environment_file["dependencies"].sort()

    # Dump yml file
    with open(out_path, "w") as f:
        yaml.dump(environment_file, f, sort_keys=False)


if __name__ == "__main__":
    fire.Fire(main)
