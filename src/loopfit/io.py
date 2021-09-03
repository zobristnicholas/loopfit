import numpy as np


def load_touchstone(file_name):
    """
    Function for loading in touchstone files. The file_name must
    point to a version 1 or 2 touchstone file.

    Args:
        file_name: string
            A file name or path to a touchstone version 1 or 2 file.

    Returns:
        f: numpy.ndarray
            The resonance frequencies in GHz.
        i: numpy.ndarray
            The in phase component of the scattering data.
        q: numpy.ndarray
            The quadrature component of the scattering data.
    """
    with open(file_name) as fid:
        values = []
        while True:
            line = fid.readline()

            # Exit the while loop if we run out of lines.
            if not line:
                break

            # Remove the comments, leading or trailing whitespace, and make
            # everything lowercase.
            line = line.split('!', 1)[0].strip().lower()

            # Skip the line if it was only comments.
            if len(line) == 0:
                continue

            # Skip the version line.
            if line.startswith('[version]'):
                continue

            # Skip the number of ports line.
            if line.startswith('[number of ports]'):
                continue

            # Skip the data order line since it's the same for all Sonnet
            # outputs.
            if line.startswith('[two-port data order]'):
                continue

            # Skip the number of frequencies line.
            if line.startswith('[number of frequencies]'):
                continue

            # skip the network data line.
            if line.startswith('[network data]'):
                continue

            # Skip the end line.
            if line.startswith('[end]'):
                continue

            # Note the options.
            if line[0] == '#':
                options = line[1:].strip().split()
                # fill the option line with the missing defaults
                options.extend(['ghz', 's', 'ma', 'r', '50'][len(options):])
                unit = options[0]
                parameter = options[1]
                if parameter != 's':
                    raise ValueError("The file must contain the S parameters.")
                data_format = options[2]
                continue

            # Collect all of the values making sure they are the right length.
            data = [float(v) for v in line.split()]
            if len(data) != 9:
                raise ValueError("The data does not come from a two port "
                                 "network.")
            values.extend(data)

    # Reshape into rows of f, s11, s21, s12, s22, s11, ...
    values = np.asarray(values).reshape((-1, 9))

    # Extract the frequency in GHz.
    multiplier = {'hz': 1.0, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9}[unit]
    f = values[:, 0] * multiplier / 1e9  # always in GHz

    # Extract the S21 parameter.
    if data_format == "ri":
        values_complex = values[:, 1::2] + 1j * values[:, 2::2]
        z = values_complex[:, 1]  # S21
    elif data_format == "ma":
        mag = values[:, 1::2]
        angle = np.pi / 180 * values[:, 2::2]
        values_complex = mag * np.exp(1j * angle)
        z = values_complex[:, 1]  # S21
    else:  # == "db"
        mag_db = values[:, 1::2]
        angle = np.pi / 180 * values[:, 2::2]
        values_complex = 10**(mag_db / 20.0) * np.exp(1j * angle)
        z = values_complex[:, 1]  # S21
    return f, z.real, z.imag
