import numpy as np
import matplotlib.pyplot as plt


class RobotUtils:
    @staticmethod
    def rotx(theta):
        """
        Create a rotation matrix around the x-axis
        :param theta: Angle in radians
        :return: Rotation matrix
        """
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])

    @staticmethod
    def roty(theta):
        """
        Create a rotation matrix around the y-axis
        :param theta: Angle in radians
        :return: Rotation matrix
        """
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])

    @staticmethod
    def rotz(theta):
        """
        Create a rotation matrix around the z-axis
        :param theta: Angle in radians
        :return: Rotation matrix
        """
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    @staticmethod
    def grotx(theta):
        """
        Create a homogenous matrix around the x-axis
        :param theta: Angle in radians
        :return: homogenous matrix
        """
        return np.array([[1, 0, 0, 0],
                         [0, np.cos(theta), -np.sin(theta), 0],
                         [0, np.sin(theta), np.cos(theta), 0],
                         [0, 0, 0, 1]])

    @staticmethod
    def groty(theta):
        """
        Create a Homogenous matrix around the y-axis
        :param theta: Angle in radians
        :return: Homogenous matrix
        """
        return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                         [0, 1, 0, 0],
                         [-np.sin(theta), 0, np.cos(theta), 0],
                         [0, 0, 0, 1]])

    @staticmethod
    def grotz(theta):
        """
        Create a Homogenous matrix around the z-axis
        :param theta: Angle in radians
        :return: Homogenous matrix
        """
        return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                         [np.sin(theta), np.cos(theta), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    @staticmethod
    def gtransl(dx, dy, dz):
        """
        Create a translation matrix
        :param dx: Translation along x-axis
        :param dy: Translation along y-axis
        :param dz: Translation along z-axis
        :return: Translation matrix
        """
        return np.array([[1, 0, 0, dx],
                         [0, 1, 0, dy],
                         [0, 0, 1, dz],
                         [0, 0, 0, 1]])


    @staticmethod
    def rotvec(v, t):
        """
        Compute the rotation matrix for rotating around an axis v by an angle t.

        Parameters:
            v (array-like): Axis of rotation (unit vector) expressed as [x, y, z].
            t (float): Angle of rotation in radians.

        Returns:
            array: Homogeneous rotation matrix.
        """
        # Normalize axis vector
        v = np.array(v)
        v = v / np.linalg.norm(v)

        # Compute components
        ct = np.cos(t)
        st = np.sin(t)
        vt = 1 - ct

        # Construct the rotation matrix
        r = np.array([[ct, -v[2]*st, v[1]*st],
                      [v[2]*st, ct, -v[0]*st],
                      [-v[1]*st, v[0]*st, ct]])

        # Construct the homogeneous rotation matrix
        r_homogeneous = np.zeros((4, 4))
        r_homogeneous[:3, :3] = np.outer(v, v) * vt + r
        r_homogeneous[:3, 3] = np.zeros(3)
        r_homogeneous[3, 3] = 1

        return r_homogeneous

    @staticmethod
    def euler_to_quaternion(phi, theta, psi):
        """
        Convert Euler angles (roll, pitch, yaw) to a quaternion
        :param phi: Roll angle in radians
        :param theta: Pitch angle in radians
        :param psi: Yaw angle in radians
        :return: Quaternion [w, x, y, z]
        """
        cy = np.cos(psi * 0.5)
        sy = np.sin(psi * 0.5)
        cp = np.cos(theta * 0.5)
        sp = np.sin(theta * 0.5)
        cr = np.cos(phi * 0.5)
        sr = np.sin(phi * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qw, qx, qy, qz])

    @staticmethod
    def quaternion_to_euler(q):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw)
        :param q: Quaternion [w, x, y, z]
        :return: Euler angles (roll, pitch, yaw)
        """
        qw, qx, qy, qz = q
        phi = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        theta = np.arcsin(2 * (qw * qy - qz * qx))
        psi = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        return phi, theta, psi

    @staticmethod
    def degrees_to_radians(degrees):
        """
        Convert degrees to radians.
        Parameters: degrees (float or array-like): Angle in degrees.
        Returns:    float or array-like: Angle in radians.
        """
        return degrees * np.pi / 180.0

    @staticmethod
    def radians_to_degrees(radians):
        """
        Convert radians to degrees.
        Parameters: radians (float or array-like): Angle in radians.
        Returns:    float or array-like: Angle in degrees.
        """
        return radians * 180.0 / np.pi

    # Function to convert UR Pose [x,y,z,Rx,Ry,Rz] to homogeneous transform matrix 4x4
    @staticmethod
    def UR_pose_to_homogeneous_transform(pose):
        # Extract position and rotation from the pose
        x, y, z = pose[:3]
        Rx, Ry, Rz = pose[3:]

        # Compute the rotation angle and normalize the rotation axis
        theta = np.linalg.norm([Rx, Ry, Rz])  # Rotation angle
        if theta == 0:
            rotation_matrix = np.eye(3)  # No rotation, identity matrix
        else:
            kx, ky, kz = Rx / theta, Ry / theta, Rz / theta  # Normalized rotation axis

            # Compute components of the rotation matrix using Rodrigues' rotation formula
            K = np.array([
                [0, -kz, ky],
                [kz, 0, -kx],
                [-ky, kx, 0]
            ])

            rotation_matrix = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

        # Construct the homogeneous transformation matrix
        homogeneous_transform = np.eye(4)
        homogeneous_transform[:3, :3] = rotation_matrix  # Set the rotation part
        homogeneous_transform[:3, 3] = [x, y, z]  # Set the translation part

        return homogeneous_transform

    @staticmethod
    def homogeneous_transform_to_UR_pose(homogeneous_transform):
        """
        Converts a homogeneous transformation matrix into a UR pose.

        :param homogeneous_transform: 4x4 homogeneous transformation matrix.
        :return: UR pose as [x, y, z, Rx, Ry, Rz].
        """
        # Extract translation components
        x, y, z = homogeneous_transform[:3, 3]

        # Extract rotation matrix
        rotation_matrix = homogeneous_transform[:3, :3]

        # Compute the rotation angle (theta) from the trace of the rotation matrix
        theta = np.arccos((np.trace(rotation_matrix) - 1) / 2)

        # Handle special case where theta is 0 (no rotation)
        if np.isclose(theta, 0):
            Rx, Ry, Rz = 0, 0, 0
        else:
            # Compute the rotation axis (normalized)
            kx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (2 * np.sin(theta))
            ky = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (2 * np.sin(theta))
            kz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (2 * np.sin(theta))

            # Scale by the rotation angle to get [Rx, Ry, Rz]
            Rx, Ry, Rz = theta * np.array([kx, ky, kz])

        # Return the UR pose
        return [x, y, z, Rx, Ry, Rz]

    # Function to convert a rotation vector [Rx,Ry,Rz] to rotation matrix 3x3
    @staticmethod
    def rotation_vector_to_matrix(rotate_vector):
        # Calculate the angle (theta) from the rotation vector
        theta = np.linalg.norm(rotate_vector)
        # If theta is very small, return the identity matrix (no rotation)
        if theta < 1e-6:
            return np.eye(3)

        # Normalize the rotation vector to get the unit vector
        u = rotate_vector / theta
        u_x, u_y, u_z = u

        # Create the skew-symmetric matrix for cross-product operation
        k = np.array([
            [0, -u_z, u_y],
            [u_z, 0, -u_x],
            [-u_y, u_x, 0]
        ])
        # Calculate the rotation matrix using Rodriguez' rotation formula
        i = np.eye(3)  # Identity matrix
        r = i + np.sin(theta) * k + (1 - np.cos(theta)) * np.dot(k, k)
        return r  # Return the resulting rotation matrix



class RobotViz:
    def __init__(self):
        self.points = []
        self.frames = []
        self.unit_vectors = []
        # Set the precision for printing floating-point values
        np.set_printoptions(precision=6)

    def visualize_coordinate_system(self, origin=np.zeros(3), x_axis=np.array([1, 0, 0]),
                                    y_axis=np.array([0, 1, 0]), z_axis=np.array([0, 0, 1])):
        """
        Visualize a coordinate system in 3D space
        :param origin: Origin of the coordinate system
        :param x_axis: Direction and magnitude of the x-axis
        :param y_axis: Direction and magnitude of the y-axis
        :param z_axis: Direction and magnitude of the z-axis
        """
        self.frames.append((origin, x_axis, y_axis, z_axis, "Coordinate System"))

    def add_point(self, point, name):
        """
        Add a point to the 3D plot
        :param point: 3D point [x, y, z]
        :param name: Name of the point
        """
        self.points.append((point, name))

    def add_frame(self, T, name):
        """
        Add a frame to the 3D plot based on a homogeneous transformation matrix
        :param T: Homogeneous transformation matrix
        :param name: Name of the frame
        """
        origin = T[:3, 3]
        x_axis = T[:3, 0]
        y_axis = T[:3, 1]
        z_axis = T[:3, 2]
        self.frames.append((origin, x_axis, y_axis, z_axis, name))

    def add_unit_vector(self, start_point, direction, name):
        """
        Add a unit vector to the 3D plot
        :param start_point: Starting point of the vector
        :param direction: Direction of the vector
        :param name: Name of the unit vector
        """
        # Normalize the direction vector to ensure its magnitude is 1
        normalized_direction = direction / np.linalg.norm(direction)
        self.unit_vectors.append((start_point, normalized_direction, name))

    def show_plots(self):
        """
        Display all stored plots
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d',  aspect='equal')

        for origin, x_axis, y_axis, z_axis, name in self.frames:
            ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', label='X-axis')
            ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', label='Y-axis')
            ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', label='Z-axis')
            ax.text(origin[0], origin[1], origin[2], name)

        for point, name in self.points:
            ax.scatter(point[0], point[1], point[2], color='k', label=name)
            ax.text(point[0], point[1], point[2], name)

        for start_point, direction, name in self.unit_vectors:
            ax.quiver(start_point[0], start_point[1], start_point[2],
                      direction[0], direction[1], direction[2], color='m', label=name)
            ax.text(start_point[0], start_point[1], start_point[2], name)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set aspect ratio to be equal
        ax.set_box_aspect([1, 1, 1])

        # Adjust axis limits based on maximum range of data along each axis
        max_range = np.array([0, 0, 0])
        for data, _ in self.points:
            max_range = np.maximum(max_range, np.max(np.abs(data), axis=0))
        for origin, x_axis, y_axis, z_axis, _ in self.frames:
            data = np.concatenate(
                (origin.reshape(1, -1), x_axis.reshape(1, -1), y_axis.reshape(1, -1), z_axis.reshape(1, -1)), axis=0)
            max_range = np.maximum(max_range, np.max(np.abs(data), axis=0))
        ax.set_xlim([-max_range[0], max_range[0]])
        ax.set_ylim([-max_range[1], max_range[1]])
        ax.set_zlim([-max_range[2], max_range[2]])

        #plt.legend()
        plt.show()

    def plot_frames(transforms):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', aspect='equal')

        # Plot frames as coordinate axes
        for i, T in enumerate(transforms):
            origin = T[:3, 3]
            x_axis = T[:3, 0]
            y_axis = T[:3, 1]
            z_axis = T[:3, 2]

            ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=1, normalize=True)
            ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=1, normalize=True)
            ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=1, normalize=True)

            # Add text labels
            ax.text(origin[0], origin[1], origin[2], f'T{i}', color='k')

            # Connect frames with lines
            if i > 0:
                prev_origin = transforms[i - 1][:3, 3]
                ax.plot([prev_origin[0], origin[0]], [prev_origin[1], origin[1]], [prev_origin[2], origin[2]], color='k')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Calculate the maximum range among all axes
        all_points = np.array([T[:3, 3] for T in transforms])
        max_range = np.max(np.ptp(all_points, axis=0))

        # Set the limits of all axes to the maximum range
        min_coords = np.min(all_points, axis=0)
        max_coords = min_coords + max_range
        ax.set_xlim([min_coords[0], max_coords[0]])
        ax.set_ylim([min_coords[1], max_coords[1]])
        ax.set_zlim([min_coords[2], max_coords[2]])
        plt.show()


class DHRobot:
    def __init__(self, dh_params):
        self.dh_params = dh_params

    def forward_kinematics_standard(self, joint_angles):
        T = np.eye(4)
        for i, (theta, d, a, alpha) in enumerate(self.dh_params):
            #theta += joint_angles[i]  # Update theta with the corresponding joint angle
            A_i = np.array([
                [np.cos(theta), -np.sin(theta), 0, a],
                [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
                [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
                [0, 0, 0, 1]
            ])
            T = np.dot(T, A_i)
        return T

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        Ts = []  # Initialize the list with the identity matrix
        for i, (theta, d, a, alpha) in enumerate(self.dh_params):
            theta += joint_angles[i]  # Update theta with the corresponding joint angle
            A_i = np.array([
                [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])
            T = np.dot(T, A_i)
            Ts.append(T.copy())  # Append the current transformation matrix to the list
        return T, Ts


