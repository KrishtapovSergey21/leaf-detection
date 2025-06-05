import xml.etree.ElementTree as ET
import os
from collections import defaultdict

def count_objects_in_xml_label(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    label_counts = defaultdict(int)
    # Iterate through each object in the XML file
    for obj in root.findall('object'):
        # Extract object label
        label = obj.find('name').text
        label_counts[label] += 1
    return label_counts

def count_instances_in_xml_label(xml_folder):
    total_instances = 0
    label_counts = defaultdict(int)
    # Iterate through each XML file in the folder
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(xml_folder, filename)
            # Count objects in the current XML file
            obj_counts = count_objects_in_xml_label(xml_path)
            # Update total instances count
            total_instances += sum(obj_counts.values())
            # Update label counts
            for label, count in obj_counts.items():
                label_counts[label] += count

    return total_instances, label_counts

if __name__ == "__main__":
    xml_folder = "E:/Work/Project/DataSet2"
    total_instances, label_counts = count_instances_in_xml_label(xml_folder)

    print("Total instances:", total_instances)
    print("Object counts by label:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
