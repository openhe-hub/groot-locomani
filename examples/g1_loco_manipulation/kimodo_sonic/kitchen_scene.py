"""A fixed preparation island inside the Adept / D4RL articulated kitchen."""

import xml.etree.ElementTree as ET

import mujoco as mj


def add_box(body, name, pos, size, material, **attrs):
    return ET.SubElement(
        body, "geom", name=name, type="box", pos=pos, size=size, material=material, **attrs
    )


def make_kitchen(robot, assets, *, soap=False):
    xml = ET.parse(robot).getroot()
    xml.find("compiler").set("meshdir", str(robot.resolve().parent / "meshes"))
    xml.find("visual/global").attrib.update(offwidth="1280", offheight="800")
    xml.find("visual/headlight").attrib.update(
        ambient=".25 .25 .26", diffuse=".38 .37 .36", specular=".12 .12 .12"
    )
    ET.SubElement(xml.find("visual"), "quality", shadowsize="3072", offsamples="4")
    xml.find("statistic").attrib.update(center="2.2 .35 1.05", extent="3.4")
    asset = xml.find("asset")
    world = xml.findall("worldbody")[-1]
    k = assets / "third_party/D4RL/d4rl/kitchen/adept_models/kitchen"
    rs = assets.parent / "robosuite/robosuite/models/assets/textures"
    for name, tex in [
        ("oak", "wood-tiles"),
        ("plaster", "cream-plaster"),
        ("walltile", "white-bricks"),
        ("timber", "light-wood"),
        ("package", "cereal"),
    ]:
        ET.SubElement(
            asset, "texture", type="2d", name=name, file=str((rs / (tex + ".png")).resolve())
        )
        ET.SubElement(
            asset,
            "material",
            name=name,
            texture=name,
            texrepeat="1 1",
            texuniform="false" if name == "package" else "true",
        )
    for name, color in [
        ("enamel", ".17 .32 .31 1"),
        ("granite", ".78 .76 .67 1"),
        ("steel", ".50 .52 .51 1"),
        ("tray", ".21 .47 .33 1"),
    ]:
        ET.SubElement(asset, "material", name=name, rgba=color)
    xml.find(".//geom[@name='floor']").set("material", "oak")
    # Full benchmark fixtures, original meshes, collision bodies, and hinge limits.
    kitchen = ET.SubElement(
        world, "body", name="kitchen_suite", pos="3.72 0.45 -.70", euler="0 0 -1.5707963"
    )
    entries = [
        ("counters", "0 0 0"),
        ("oven", "0 0 0"),
        ("backwall", "0 0 0"),
        ("slidecabinet", ".4 .30 2.45"),
        ("hingecabinet", "-.504 .28 2.45"),
        ("kettle", "-.18 .16 1.63"),
        ("microwave", None),
    ]
    for name, pos in entries:
        module = ET.parse(k / "assets" / (name + "_asset.xml")).getroot()
        for a in module.findall("asset/*"):
            if "file" in a.attrib:
                a.set("file", str((k / a.get("file")).resolve()))
            asset.append(a)
        for a in (
            module.findall("default")
            if module.find("default").get("class")
            else module.findall("default/default")
        ):
            default = xml.find("default")
            if default is None:
                default = ET.SubElement(xml, "default")
            default.append(a)
        body = (
            ET.SubElement(kitchen, "body", name=name + "_fixture", pos=pos)
            if pos
            else ET.SubElement(
                world, "body", name="appliance", pos="2.54 -.22 .77", euler="0 0 -1.5707963"
            )
        )
        for a in ET.parse(k / "assets" / (name + "_chain.xml")).getroot():
            body.append(a)
    add_box(world, "back_wall", "4.65 .45 1.65", ".08 3.95 1.65", "plaster")
    add_box(world, "side_wall", "1.1 3.2 1.65", "3.5 .07 1.65", "walltile")
    add_box(world, "skirting", "1.1 3.11 .055", "3.5 .035 .055", "timber")
    add_box(world, "island_base", "2.72 .20 .34", ".50 .96 .34", "enamel")
    add_box(world, "island_worktop", "2.68 .20 .735", ".61 1.01 .035", "granite")
    for j, y in enumerate([-0.42, 0.20, 0.82]):
        add_box(world, f"front{j}", "2.208 " + str(y) + " .38", ".016 .292 .286", "timber")
        add_box(world, f"handle{j}", "2.178 " + str(y) + " .57", ".018 .105 .012", "steel")
    for j, z in enumerate([1.20, 1.66]):
        add_box(world, f"wall_shelf{j}", f"1.5 2.98 {z}", "1.0 .15 .025", "timber")
        for i in range(5):
            ET.SubElement(
                world,
                "geom",
                name=f"canister{j}_{i}",
                type="cylinder",
                pos=f"{0.85 + i * 0.29} 2.96 {z + 0.115}",
                size=".065 .09",
                rgba=(".67 .42 .24 1" if i % 2 else ".83 .78 .65 1"),
            )
    add_box(
        world,
        "receiving_mat",
        "2.46 .78 .773",
        ".13 .19 .003",
        "tray",
        contype="0",
        conaffinity="0",
    )
    carton = ET.SubElement(world, "body", name="carton", pos="2.17 .78 .895")
    ET.SubElement(carton, "freejoint", name="carton_free")
    add_box(
        carton,
        "cereal_carton",
        "0 0 0",
        ".042 .080 .125",
        "package",
        mass=".32",
        friction=".45 .01 .001",
    )
    # Scanned Robosuite groceries retain their original UVs and printed packaging.
    mesh_dir = rs.parent / "objects/meshes"
    for item, scale, tex in [
        ("cereal", "1.60 2.8 1.667", "cereal"),
        ("bread", "2.4 2.4 1.6", "bread"),
        ("lemon", "1.8 1.8 1.8", "lemon"),
        ("can", "1.35 1.35 1.45", "can"),
    ]:
        ET.SubElement(
            asset,
            "mesh",
            name="food_" + item,
            file=str((mesh_dir / (item + ".msh")).resolve()),
            scale=scale,
        )
        ET.SubElement(
            asset,
            "texture",
            name="label_" + item,
            type="2d",
            file=str((rs / (tex + ".png")).resolve()),
        )
        ET.SubElement(
            asset,
            "material",
            name="food_" + item,
            texture="label_" + item,
            specular=".12",
            shininess=".12",
        )
    carton.find("geom").set("group", "4")
    ET.SubElement(
        carton,
        "geom",
        type="mesh",
        mesh="food_cereal",
        material="food_cereal",
        euler="0 0 -1.5707963",
        mass="0",
        contype="0",
        conaffinity="0",
    )
    for item, pos in [
        ("bread", "2.86 .42 .819"),
        ("lemon", "2.96 .58 .806"),
        ("lemon", "2.99 .49 .806"),
        ("can", "2.99 -.68 .83"),
        ("can", "2.91 -.72 .83"),
    ]:
        ET.SubElement(
            world,
            "geom",
            type="mesh",
            mesh="food_" + item,
            material="food_" + item,
            pos=pos,
            contype="0",
            conaffinity="0",
        )
    # Reduce specular clipping from the original arm benchmark's close-range task lights.
    for parent in xml.iter():
        for light in list(parent.findall("light")):
            if parent.tag != "default":
                parent.remove(light)
    for a in xml.findall(".//material"):
        a.set("reflectance", "0")
        a.set("specular", ".16")
    for key, tint in [
        ("oak", ".48 .44 .40 1"),
        ("plaster", ".71 .73 .72 1"),
        ("walltile", ".72 .73 .74 1"),
        ("timber", ".70 .57 .47 1"),
    ]:
        xml.find(".//material[@name='" + key + "']").set("rgba", tint)
    ET.SubElement(
        world,
        "light",
        pos="-.5 -2.5 4.3",
        dir=".55 .34 -.78",
        ambient=".03 .03 .03",
        diffuse=".57 .55 .52",
        specular=".13 .13 .12",
        directional="true",
    )
    ET.SubElement(
        world,
        "light",
        pos="2 2 2.7",
        dir=".1 -.7 -.6",
        diffuse=".22 .24 .26",
        specular="0 0 0",
        castshadow="false",
        directional="true",
    )
    if soap:
        bottle = ET.SubElement(world, "body", name="soap_dispenser", pos="2.17 .43 .77")
        ET.SubElement(
            bottle, "geom", type="cylinder", size=".055 .015", pos="0 0 .015", rgba=".15 .22 .22 1"
        )
        ET.SubElement(
            bottle, "geom", type="capsule", size=".050 .068", pos="0 0 .118", rgba=".08 .44 .42 1"
        )
        ET.SubElement(
            bottle, "geom", type="cylinder", size=".033 .019", pos="0 0 .225", material="steel"
        )
        head = ET.SubElement(bottle, "body", name="pump_head", pos="0 0 .27")
        ET.SubElement(
            head,
            "joint",
            name="pump_slide",
            type="slide",
            axis="0 0 -1",
            range="0 .038",
            solreflimit=".005 1",
            solimplimit=".96 .99 .001",
            stiffness="90",
            springref="-.010",
            damping="2",
            limited="true",
        )
        ET.SubElement(
            head,
            "geom",
            type="cylinder",
            size=".013 .035",
            pos="0 0 -.007",
            material="steel",
            mass=".018",
        )
        ET.SubElement(
            head, "geom", type="ellipsoid", size=".061 .043 .012", rgba=".91 .89 .81 1", mass=".045"
        )
        ET.SubElement(
            head,
            "geom",
            type="capsule",
            fromto="0 0 0 0 -.081 -.012",
            size=".012",
            rgba=".9 .88 .80 1",
            mass=".012",
        )
        ET.SubElement(head, "site", name="pump_surface", pos="-.013 0 .012", size=".004")
        # The stem slides inside the collar. Both retain collisions with the hand.
        for geom in bottle.findall("geom"):
            geom.attrib.update(contype="2", conaffinity="1")
        for geom in head.findall("geom"):
            geom.attrib.update(contype="4", conaffinity="1")
        add_box(
            world,
            "sink_sponge",
            "2.17 .337 .788",
            ".063 .029 .018",
            "granite",
            rgba=".77 .64 .28 1",
        )
    return mj.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
