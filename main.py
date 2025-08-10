from pioneer_sdk import Pioneer, Camera

from flight_utils import load_flight_coordinates, get_config
from drone_navigation import FlightMissionRunner








if __name__ == "__main__":
    try:
        flight_height = 1.5

        MAP_POINTS = load_flight_coordinates()


        pioneer_conf = get_config('local')  # или 'global'   local

        # Инициализация миссии
        mission = FlightMissionRunner(MAP_POINTS)
        
        # Инициализация дрона
        pioneer = Pioneer(
            name="pioneer",
            ip=pioneer_conf["ip"],
            mavlink_port=pioneer_conf["port"],
            connection_method="udpout",
            device="dev/serial0",
            baud=115200, log_connection=True, logger=True
        )

        

        pioneer.arm()
        pioneer.takeoff()
        
        # Начало миссии
        first_point = mission.get_next_point()
        x, y, z = first_point
        pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
        

        # Основной цикл миссии
        while not mission.is_complete():
            # time.sleep(0.01)



            if pioneer.point_reached():
                # Переход к следующей точке
                next_point = mission.get_next_point()
                if next_point:
                    x, y, z = next_point
                    pioneer.go_to_local_point(x=x, y=y, z=flight_height, yaw=0)
            

           

    except Exception as e:
        print(f"Ошибка обработки: {str(e)}")
    
              

    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания (Ctrl+C)")
    
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
    
    finally:
        print("Завершение работы")
        pioneer.land()
        pioneer.disarm()
        pioneer.close_connection()
