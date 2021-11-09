import pathlib
import logging
import numpy as np

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def load_touchstone(file_name, component=(2, 1)):
    """
    Function for loading a scattering matrix element from a touchstone
    file. The file_name must point to a version 1 or 2 touchstone file.

    Args:
        file_name: string
            A file name or path to a touchstone version 1 or 2 file.
        component: 2-tuple of integers
            The (row, column) of the scattering matrix element.

    Returns:
        f: numpy.ndarray
            The resonance frequencies in GHz.
        i: numpy.ndarray
            The in-phase (real) component of the scattering data.
        q: numpy.ndarray
            The quadrature (imaginary) component of the scattering data.
    """
    # Get the number of ports from the extension (version 1)
    extension = pathlib.Path(file_name).suffix
    if (extension[1] == 's') and (extension[-1] == 'p'):  # sNp
        try:
            n_ports = int(extension[1:-1])
            version = 1.
        except ValueError:
            message = ("The file name does not have a s-parameter extension. "
                       f"It is [{extension}] instead. Please, correct the "
                       "extension to the form: 'sNp', where N is any integer.")
            raise IOError(message)
    elif extension == '.ts':
        n_ports = None
    else:
        message = ('The filename does not have the expected Touchstone '
                   'extension (.sNp or .ts)')
        raise IOError(message)

    values = []
    flip_port_order = False
    matrix_format = 'full'
    with open(file_name) as fid:
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
                version = float(line.partition('[version]')[2])
                continue

            # Grab the number of ports if it exists.
            if line.startswith('[number of ports]'):
                n_ports = int(line.partition('[number of ports]')[2])
                continue

            # Skip the data order line since it's the same for all Sonnet
            # outputs.
            if line.startswith('[two-port data order]'):
                order = line.partition('[two-port data order]')[2].strip()
                if order == '21_12':
                    flip_port_order = True
                continue

            # Skip the number of frequencies line.
            if line.startswith('[number of frequencies]'):
                continue

            if line.startswith('[matrix format]'):
                matrix_format = line.partition('[matrix format]')[2].strip()
                continue

            if line.startswith('[mixed-mode order]'):
                message = "The mixed-mode order data format is not supported."
                raise IOError(message)

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

            # Collect all of the values.
            values.extend([float(v) for v in line.split()])

    # Version 1 files have weird port order for 2 port matricies
    if version < 2 and n_ports == 2:
        flip_port_order = True

    # Reshape into rows of f, s11, s12, s13, s21, s22, s23, ...
    if matrix_format == 'full':
        values = np.asarray(values).reshape((-1, 2 * n_ports**2 + 1))
    else:  # lower or upper
        values = np.asarray(values).reshape((-1, n_ports**2 + n_ports + 1))

    # Remove noise values
    noise = np.where(np.diff(values[:, 0]) < 0)[0]  # f should increase
    if len(noise) != 0:
        values = values[noise[0] + 1:, :]

    # Extract the frequency in GHz.
    multiplier = {'hz': 1.0, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9}[unit]
    f = values[:, 0] * multiplier / 1e9  # always in GHz

    # Get the index of the desired component.
    row, column = component
    if flip_port_order:
        row, column = column, row
    if matrix_format == 'full':
        index = (row - 1) * n_ports + column
    elif matrix_format == 'lower':
        if row < column:
            row, column = column, row
        index = row * (row - 1) / 2 + column
    else:  # 'upper'
        if row > column:
            row, column = column, row
        index = (n_ports * (n_ports - 1) / 2
                 - (n_ports - row + 1) * (n_ports - row) / 2 + column)
    index = index * 2 - 1  # components come in pairs

    # Extract the S21 parameter.
    if data_format == "ri":
        z = values[:, index] + 1j * values[:, index + 1]
    elif data_format == "ma":
        mag = values[:, index]
        angle = np.pi / 180 * values[:, index + 1]
        z = mag * np.exp(1j * angle)
    else:  # == "db"
        mag_db = values[:, index]
        angle = np.pi / 180 * values[:, index + 1]
        z = 10**(mag_db / 20.0) * np.exp(1j * angle)
    return f, z.real, z.imag
