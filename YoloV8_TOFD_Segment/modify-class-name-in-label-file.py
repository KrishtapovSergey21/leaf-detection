import xml.etree.ElementTree as ET
import os
def modify_xml_label(xml_file, new_label):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Modify the label in each object
    for obj in root.findall('object'):
        # Find the name element and update its text
        name_element = obj.find('name')
        if name_element is not None:
            name_element.text = new_label
    # Write the modified XML back to a file
    tree.write(xml_file)


if __name__ == "__main__":
    xml_folder = "E:/Work/Project/DataSet-Detection"
    new_label = "fault"
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(xml_folder, filename)
            modify_xml_label(xml_path, new_label)



