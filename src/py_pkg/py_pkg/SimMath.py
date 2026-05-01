#Copied from divetest files: depth_control_node_SimMath.py

import numpy as np


pi = np.pi



class Vector:
    """
    Represents a vector in three-dimensional space.

    Wrapper around a numpy.ndarray

    Attributes:
        vec (numpy.ndarray): A numpy array representing the vector's coordinates in x, y, and z directions.

    Methods:
        __init__(self, x: float = 0, y: float = 0, z: float = 0): Initializes a Vector object with the given coordinates.
        x(self) -> float: Returns the x-coordinate value of the vector.
        y(self) -> float: Returns the y-coordinate value of the vector.
        z(self) -> float: Returns the z-coordinate value of the vector.
    """

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        """
        Initializes a Vector object with the given coordinates.

        Args:
            x (float): The x-coordinate value (default is 0).
            y (float): The y-coordinate value (default is 0).
            z (float): The z-coordinate value (default is 0).
        """

        self.vec : np.ndarray = np.array([x, y, z])



    def x(self, set: float | int | None = None) -> float:
        """
        Returns the x-coordinate value of the vector.

        Returns:
            float: The x-coordinate value.
        """

        if set is not None:
            self.vec[0] = set

        return self.vec[0]



    def y(self, set: float | int | None = None) -> float:
        """
        Returns the y-coordinate value of the vector.

        Returns:
            float: The y-coordinate value.
        """

        if set is not None:
            self.vec[1] = set

        return self.vec[1]



    def z(self, set: float | int | None = None) -> float:
        """
        Returns the z-coordinate value of the vector.

        Returns:
            float: The z-coordinate value.
        """

        if set is not None:
            self.vec[2] = set

        return self.vec[2]
    


    def rotate(self, angle: float, axis: 'Vector') -> 'Vector':
        """
        Rotates the vector by the given angle around the specified axis.

        Args:
            angle (float): The angle in radians.
            axis (Vector): The axis of rotation.

        Returns:
            Vector: A new Vector instance representing the rotated vector.
        """

        # Normalize the axis vector
        normalized_axis = axis.normalized()

        # Compute the sine and cosine of the angle
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Compute the cross product between the axis and the vector
        cross_product = normalized_axis.cross(self)

        # Compute the dot product between the axis and the vector
        dot_product = normalized_axis.dot(self)

        # Compute the rotated vector
        rotated_vec = self * cos_angle + cross_product * sin_angle + normalized_axis * dot_product * (1 - cos_angle)

        return rotated_vec
    


    def magnitude(self) -> float:
        return float(np.linalg.norm(self.vec))

    

    def normalized(self) -> 'Vector':
        """
        Returns a normalized version of the vector.

        Returns:
            Vector: A new Vector instance with normalized coordinates.
        """

        norm = np.linalg.norm(self.vec)

        if norm == 0:
            return Vector(0, 0, 0)
        
        normalized_vec = self.vec / norm

        return Vector(normalized_vec[0], normalized_vec[1], normalized_vec[2])
    


    def modulo(self, val: float) -> 'Vector':

        return Vector(self.x() % val, self.y() % val, self.z() % val)
    


    def loop(self, low: float, high: float) -> 'Vector':
        """
        Loops the vector's coordinates between the specified low and high values.

        Args:
            low (float): The lower bound value.
            high (float): The upper bound value.

        Returns:
            Vector: A new Vector instance with looped coordinates.
        """

        # return self.modulo(high - low) + Vector(low, low, low)
        range_size = high - low

        looped_x = ((self.x() - low) % range_size) + low
        looped_y = ((self.y() - low) % range_size) + low
        looped_z = ((self.z() - low) % range_size) + low

        return Vector(looped_x, looped_y, looped_z)
        
    


    def dot(self, other: 'Vector') -> float:
        return np.dot(self.vec, other.vec)
    


    def cross(self, other: 'Vector') -> 'Vector':
        """
        Computes the cross product of two vectors.

        Args:
            other (Vector): Another vector to compute the cross product with.

        Returns:
            Vector: A new Vector instance representing the cross product of the two vectors.
        """


        cross_product = np.cross(self.vec, other.vec)
        return Vector(cross_product[0], cross_product[1], cross_product[2])
    


    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Performs component-wise addition with another vector.

        Args:
            other (Vector): Another vector to perform addition with.

        Returns:
            Vector: A new Vector instance representing the result of the addition.
        """

        added_vec = self.vec + other.vec
        return Vector(added_vec[0], added_vec[1], added_vec[2])
    



    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Performs component-wise subtraction with another vector.

        Args:
            other (Vector): Another vector to perform subtraction with.

        Returns:
            Vector: A new Vector instance representing the result of the subtraction.
        """

        subtracted_vec = self.vec - other.vec
        return Vector(subtracted_vec[0], subtracted_vec[1], subtracted_vec[2])
    



    def __mul__(self, scalar: int | float) -> 'Vector':
        """
        Multiply the vector by a scalar or perform dot product with another vector.

        Parameters:
            scalar (int | float | None): The scalar value to multiply the vector by. Defaults to None.
            other (Vector | None): Another vector to perform dot product with. Defaults to None.

        Returns:
            Vector | float | None: If scalar is provided, returns a new Vector object representing the result of the multiplication.
            If other is provided, returns the dot product of the two vectors as a float.
            If neither scalar nor other is provided, returns None.
        """

        multiplied_vec = self.vec * scalar
        return Vector(multiplied_vec[0], multiplied_vec[1], multiplied_vec[2])
    



    def __truediv__(self, scalar: int | float) -> 'Vector':
        """
        Performs scalar division on the vector.

        Args:
            scalar (int or float): The scalar value to divide the vector by.

        Returns:
            Vector: A new Vector instance representing the result of the scalar division.
        """

        divided_vec = self.vec / scalar

        return Vector(divided_vec[0], divided_vec[1], divided_vec[2])
    


    def __neg__(self) -> 'Vector':
        """
        Performs unary negation on the vector.

        Returns:
            Vector: A new Vector instance representing the negation of the vector.
        """

        negated_vec = -self.vec

        return Vector(negated_vec[0], negated_vec[1], negated_vec[2])
    


    def __str__(self) -> str:
        """
        Returns a string representation of the Vector object.

        Returns:
            str: A string representation of the Vector object.
        """

        return f"Vector({self.vec[0]}, {self.vec[1]}, {self.vec[2]})"

    



def euler_to_direction(roll: float, pitch: float, yaw: float) -> 'Vector':
    """
    Converts Euler angles to a direction vector.

    Args:
        roll (float): The roll angle in radians.
        pitch (float): The pitch angle in radians.
        yaw (float): The yaw angle in radians.

    Returns:
        Vector: The direction vector.
    """

    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)
    return Vector(x, y, z)



def euler_to_rotation_matrix(euler_angles: Vector) -> np.ndarray:
    """
    Converts Euler angles to a rotation matrix.

    Args:
        roll (float): The roll angle in radians.
        pitch (float): The pitch angle in radians.
        yaw (float): The yaw angle in radians.

    Returns:
        np.ndarray: The rotation matrix.
    """
    roll = euler_angles.x()
    pitch = euler_angles.y()
    yaw = euler_angles.z()

    # Rotation matrix for roll
    R_x = np.array([
        [1,             0,             0],
        [0,  np.cos(roll), -np.sin(roll)],
        [0,  np.sin(roll),  np.cos(roll)]
    ])

    # Rotation matrix for pitch
    R_y = np.array([
        [ np.cos(pitch),   0,  np.sin(pitch)],
        [             0,   1,              0],
        [-np.sin(pitch),   0,  np.cos(pitch)]
    ])

    # Rotation matrix for yaw
    R_z = np.array([
        [ np.cos(yaw), -np.sin(yaw),    0],
        [ np.sin(yaw),  np.cos(yaw),    0],
        [           0,            0,    1]
    ])

    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R




def lerp(start: float, end: float, t: float) -> float:
    """
    Linearly interpolates between two values.

    Args:
        start (float): The starting value.
        end (float): The ending value.
        t (float): The interpolation parameter. Should be between 0 and 1.

    Returns:
        float: The interpolated value.

    """

    return start + ((end - start) * t)





def clamp(val: float, min: float, max: float) -> float:
    """
    Clamps a value between a minimum and maximum value.

    Args:
        val (float): The value to be clamped.
        min (float): The minimum value.
        max (float): The maximum value.

    Returns:
        float: The clamped value.

    """

    if val > max:
        return max

    if val < min:
        return min

    return val






def clamp_mag(val: float, max_mag: float) -> float:
    """
    Clamps the value `val` between -max_mag and max_mag.

    Args:
        val (float): The value to be clamped.
        max_mag (float): The maximum magnitude allowed.

    Returns:
        float: The clamped value.

    """
    
    if val > max_mag:
        return max_mag

    if val < -max_mag:
        return -max_mag

    return val





def sign(val: int | float) -> int:
    """
    Returns the sign of a given value.

    Args:
        val (int or float): The value to determine the sign of.

    Returns:
        int: 1 if the value is positive, 0 if the value is zero, -1 if the value is negative.
    """
    if val > 0:
        return 1
    
    if val == 0:
        return 0
    
    return -1
