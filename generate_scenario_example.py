from simply import scenario

if __name__ == "__main__":
    sc = scenario.create_random(12, 10)
    print(sc.to_dict())

    sc.power_network.update_shortest_paths()
    print(sc.power_network.short_paths)

    sc.power_network.plot()
    sc.power_network.to_image(10, 7)
    sc.power_network.to_json()
    print(sc)
