import subprocess

from ml_dev.environment import TFLITE_MODEL_FILE, C_ARRAY_MODEL_FILE


def main() -> None:
    process = subprocess.run(["xxd", "-i", str(TFLITE_MODEL_FILE)], capture_output=True)
    output_lines = process.stdout.decode("utf8").splitlines(keepends=True)

    with open(C_ARRAY_MODEL_FILE, "w") as out_file:
        out_file.writelines(
            f"const {line}" if line.startswith("unsigned") else line
            for line in output_lines
        )


if __name__ == "__main__":
    main()
