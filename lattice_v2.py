"""
A simple model of mechanical waves in a solid medium.
"""
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple
from vpython import canvas, color, pyramid, rate, sphere, sqrt, vector

#############################################
# MODEL SETUP

# Physical constants used in the simulation #
SIM_TIME = 2  # The simulation length (in s)
dt = 0.0025  # The timestep to use (s)

# Constanst determining parameters of the lattice
ELAS = 70 * 10000000  # spring constant
DIM = (20, 20, 1)  # Number of nodes in the lattice
LAT_CONST = 5  # The spacing between nodes in the lattice
RAD = 0.15 * LAT_CONST  # atom radius
MASS = 2700 * LAT_CONST ** 3  # Mass for a cubic metre of granite
# Color of the corner atoms (Set to None not to mark corners)
CORNER_COLOR = color.red
CORNERS = {
    (i, j, k)
    for i in (-round(DIM[0] / 2), DIM[0] - round(DIM[0] / 2) - 1)
    for j in (-round(DIM[1] / 2), DIM[1] - round(DIM[1] / 2) - 1)
    for k in (-round(DIM[2] / 2), DIM[2] - round(DIM[2] / 2) - 1)
}

# Parameters for the out-of-place atom which starts the vibration in the lattice
# Start the vibration on the bottom layer
SEEDS = [
    {
        "pos": (-round(DIM[0] / 2) + 3, j, -round(DIM[2] / 2)),
        "disp": (0.0, -0.0, 0.5),
        "col": color.red,
    }
    for j in (-3,)
]
_max_disp = sqrt(max([sum((x["disp"][i] ** 2 for i in (0, 1, 2))) for x in SEEDS]))
_min_disp = sqrt(min([sum((x["disp"][i] ** 2 for i in (0, 1, 2))) for x in SEEDS]))

# Parameters for a sensor measuring movement in the medium
# Location of the motion sensor
# Note that this is always on the top layer
SENSOR_LOC = (0, 0, DIM[2] - round(DIM[2] / 2) - 1)
# Colour of the motion sensor
SENSOR_COLOR = color.green
SENSOR_MASS = MASS

# Camera setup
SCENE_CENTER = vector(0, 0, LAT_CONST)
SCENE_FORWARD = vector(0, 1, -0.5)
COLOUR_DIMENSIONS = (2,)
#############################################
# Warnings for incorrect placement
for loc in (*(x["pos"] for x in SEEDS), SENSOR_LOC):
    for i in (0, 1, 2):
        if loc[i] not in range(-round(DIM[i] / 2), DIM[i] - round(DIM[i] / 2)):
            raise IOError(
                """
        Please ensure that the seed location 'SEED_LOC' and the
        sensor location 'SENSOR_LOC' are located with in the lattice.
        Note that the lattice is symetric around (0,0,0)
        """
            )
#############################################


def compute_forces_on_atom(
    atom_idx: Tuple[int], lattice: Dict[Tuple[int], sphere]
) -> Tuple[Tuple[int], vector]:
    """Computes all the forces acting on an atom and returns

    :param atom_idx: The atom to compute the complete forces on (in lattice coordinates)
    :type atom_idx: Tuple[int]
    :param lattice: The lattice dictionary. Here the keys are locations (in lattice coordinates);
        the values are the spheres representing the "atoms"
    :type lattice: Dict[Tuple[int], sphere]
    :return: A tuple of the atom location (in lattice index coordinates) and the force experienced by this atom
    :rtype: Tuple[Tuple[int], vector]
    """

    # print('Computing forces on atom: {}'.format(atom_idx))
    forces = vector(0, 0, 0)
    for neighbour_atom in get_neighbours(atom_idx, lattice):
        atom = lattice[atom_idx]
        disp = neighbour_atom.pos - atom.pos
        rest_disp = neighbour_atom.rest_pos - atom.rest_pos

        if disp.mag2 != rest_disp.mag2:
            extension = sqrt(disp.mag2) - sqrt(rest_disp.mag2)
            forces += (
                (1 if extension >= 0 else -1) * 0.5 * ELAS * extension ** 2 * disp.hat
            )

    return forces


def apply_force_to_atom(atom: sphere, force: vector) -> None:
    """Applies the specified force to the specified atom.

    :param atom: The "atom" upon which the force is acting
    :type atom: sphere
    :param force: The force applied to the atom
    :type force: vector
    """
    # Update the locations of all the atoms except those in the corner
    if atom.type != "corner":
        atom.mom += force * dt
        atom.pos += atom.mom * dt / atom.M
        diff = vector(0, 0, 0)
        if atom.type == "default":
            diff = atom.pos - atom.rest_pos
            atom.color = vector(0.5, 0.5, 0.5) + 5 * diff


def create_default_atom(pos: Tuple[int], **kwargs) -> dict:
    """A factory which returns default atom parameters for an atom/node in the lattice

    :param rest_pos: The rest position of the node in question
    :type rest_pos: Tuple[int]
    :param kwargs: Additional keyword arguments used to overwrite the defaults
    :return: a dictionary describing the parameters of a node in the lattice
    :rtype: dict
    """
    default_node = {
        "pos": vector(*map(lambda x: LAT_CONST * x, pos)),
        "radius": RAD,
        "M": MASS,
        "mom": vector(0, 0, 0),
        "rest_pos": vector(*map(lambda x: LAT_CONST * x, pos)),
        "color": vector(0.5, 0.5, 0.5),
        "type": "default",
    }
    for k, v in kwargs.items():
        default_node[k] = v

    return default_node


def create_lattice(freeze_corners: bool = True) -> Dict[Tuple[int], sphere]:
    """Creates a lattice for the simulation according to the default parametrs defined at the top of the script

    :return: The lattice dictionary
    :rtype: Dict[Tuple[int], sphere]
    """

    lattice_parms = dict()
    # Determine the corners of the lattice

    # Note where the corners are
    if freeze_corners:
        lattice_parms.update(
            {
                loc: create_default_atom(loc, color=CORNER_COLOR, type="corner")
                for loc in CORNERS
            }
        )

    lattice = dict()
    # Note where the sensor is
    lattice[SENSOR_LOC] = pyramid(
        pos=LAT_CONST * vector(*SENSOR_LOC),
        size=vector(1, 1, 2),
        length=0.5 * LAT_CONST,
        radius=RAD,
        color=SENSOR_COLOR,
        axis=vector(0, 0, 1),
        type="sensor",
        timestamp=list(),
        displacements=list(),
        rest_pos=LAT_CONST * vector(*SENSOR_LOC),
        mom=vector(0, 0, 0),
        M=SENSOR_MASS,
    )
    # Note where the seeds are
    lattice.update(
        {
            x["pos"]: sphere(
                pos=LAT_CONST * (vector(*x["pos"]) + vector(*x["disp"])),
                radius=2 * RAD,
                color=x["col"],
                type="seed",
                rest_pos=LAT_CONST * vector(*x["pos"]),
                mom=vector(0, 0, 0),
                M=MASS,
            )
            for x in SEEDS
        }
    )

    lattice.update(
        {
            (i, j, k): sphere(
                **lattice_parms.get((i, j, k), create_default_atom((i, j, k)))
            )
            for i in range(-round(DIM[0] / 2), DIM[0] - round(DIM[0] / 2))
            for j in range(-round(DIM[1] / 2), DIM[1] - round(DIM[1] / 2))
            for k in range(-round(DIM[2] / 2), DIM[2] - round(DIM[2] / 2))
            if (i, j, k) not in lattice
        }
    )

    return lattice


def get_neighbours(
    atom_loc: Tuple[int], lattice: Dict[Tuple[int], sphere], closest_only: bool = False
) -> List[sphere]:
    """Get's the neighbours of a given atom location in the given lattice

    :param atom_loc: The location of the "atom" in question in lattice index coordinates
    :type atom_loc: Tuple[int]
    :param lattice: The lattice in which the seed atom
    :type lattice: Dict[Tuple[int], sphere]
    :param closest_only: Whether to consider only the neighbours of minimum distance away,
        defaults to False
    :type closest_only: bool, optional
    :return: A list of the neighbours of the given "atom" coordinates
    :rtype: List[sphere]
    """

    if closest_only:
        pot_ngh_locs = [
            tuple(atom_loc[i] + x[i] for i in range(3))
            for x in [
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ]
        ]
    else:
        pot_ngh_locs = [
            (atom_loc[0] + i, atom_loc[1] + j, atom_loc[2] + k)
            for i in range(-1, 2)
            for j in range(-1, 2)
            for k in range(-1, 2)
        ]

    neighbours = [lattice[x] for x in pot_ngh_locs if x in lattice and x != atom_loc]
    return neighbours


def run_sim():

    # Create the canvas for the scene of the mechanical wave simulation
    scene = canvas(
        title="Mechanical wave simulation",
        width=800,
        height=600,
        center=SCENE_CENTER,
        forward=SCENE_FORWARD,
    )

    # Initialise the lattice
    lattice = create_lattice(freeze_corners=False)
    t = 0  # Initial time for simulation

    while t < SIM_TIME:
        rate(int(1 / dt))

        # Calculate and apply the forces throughout the model
        forces = {idx: compute_forces_on_atom(idx, lattice) for idx in lattice}
        for atom_idx, force in forces.items():
            apply_force_to_atom(lattice[atom_idx], force)

        # record the information with the sensor
        sensor = lattice[SENSOR_LOC]
        sensor.timestamp.append(t)
        sensor.displacements.append(sensor.pos - sensor.rest_pos)

        if int(t - dt) != int(t):
            print("t = {}".format(t))
        t += dt

    # Plot displacements for the sensor
    plt.title("Sensor's Displacement in Three Dimensions", fontweight=1000)
    plt.xlabel("Time (s)", fontweight=1000)
    plt.ylabel("Displacement (m)", fontweight=1000)
    x_disp = plt.plot(
        sensor.timestamp,
        [disp.x for disp in sensor.displacements],
        label="x displacement",
    )
    y_disp = plt.plot(
        sensor.timestamp,
        [disp.y for disp in sensor.displacements],
        label="y displacement",
    )
    z_disp = plt.plot(
        sensor.timestamp,
        [disp.z for disp in sensor.displacements],
        label="z displacement",
    )
    plt.legend(title="Dimension:")

    plt.show()
