import rclpy
import DR_init

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 30, 30
CYCLE = 3

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("rokey_move", namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import (
            set_tool,
            set_tcp,
            movej,
            movel,
        )

        from DR_common2 import posx, posj

    except ImportError as e:
        node.get_logger().info(f"Error importing DSR_ROBOT2 : {e}")
        return

    set_tool("Tool Weight_1")
    set_tcp("GripperDA_v1")

    homej = posj([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
    posj1 = posj([0.0, 0.0, 90.0, 0.0, 30.0, 0.0])

    posx1 = posx([350.0, 34.5, 350.0, 45.0, 180.0, 45.0])
    posx2 = posx([350.0, 34.5, 300.0, 45.0, 180.0, 45.0])

    movej(homej, vel=VELOCITY, acc=ACC)

    try:
        for i in range(CYCLE):
            node.get_logger().info(f"Cycle {i+1}")

            node.get_logger().info(f"Moving to joint position: {posj1}")
            movej(posj1, vel=VELOCITY, acc=ACC)

            node.get_logger().info(f"Moving to task position: {posx1}")
            movel(posx1, vel=VELOCITY, acc=ACC)

            node.get_logger().info(f"Moving to task position: {posx2}")
            movel(posx2, vel=VELOCITY, acc=ACC)

    except KeyboardInterrupt:
        node.get_logger().info("Program Stopped")

    except Exception as e:
        node.get_logger().info(f"Robot Error: {e}")

    finally:
        movej(homej, vel=VELOCITY, acc=ACC)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
