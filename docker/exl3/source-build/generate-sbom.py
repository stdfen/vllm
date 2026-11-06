"""Create a package-level CycloneDX inventory and validate its pinned schema."""

import argparse
import datetime
import json
import pathlib
import re
import urllib.parse
import uuid

import jsonschema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=pathlib.Path, required=True)
    parser.add_argument("--image-inspect", type=pathlib.Path, required=True)
    parser.add_argument("--schema", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    image = json.loads(args.image_inspect.read_text())[0]
    components = []
    for name, version in sorted(inventory["pythonPackages"].items()):
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        purl = f"pkg:pypi/{normalized}@{urllib.parse.quote(version, safe='')}"
        components.append(
            {
                "type": "library",
                "name": name,
                "version": version,
                "bom-ref": purl,
                "purl": purl,
            }
        )
    for line in inventory["systemPackages"]:
        binary, version = line.split("\t", 1)
        name, separator, architecture = binary.partition(":")
        purl = (
            f"pkg:deb/ubuntu/{urllib.parse.quote(name, safe='')}"
            f"@{urllib.parse.quote(version, safe='')}"
        )
        if separator:
            purl += "?arch=" + urllib.parse.quote(architecture, safe="")
        components.append(
            {
                "type": "library",
                "name": binary,
                "version": version,
                "bom-ref": purl,
                "purl": purl,
            }
        )
    refs = [component["bom-ref"] for component in components]
    if len(set(refs)) != len(refs):
        raise RuntimeError("Duplicate normalized package identity")
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": "urn:uuid:" + str(uuid.uuid4()),
        "version": 1,
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "component": {
                "type": "container",
                "name": image["RepoTags"][0],
                "bom-ref": image["Id"],
                "version": image["Id"],
            },
            "properties": [
                {
                    "name": "ai.vllm.inventory.coverage",
                    "value": "Installed Python and Debian packages; excludes full "
                    "vendored dependency graph, file hashes, host GPU driver "
                    "and models. "
                    "Post-install pruning evidence is stored alongside this BOM.",
                }
            ],
        },
        "components": components,
        "dependencies": [{"ref": image["Id"], "dependsOn": refs}],
    }
    schema = json.loads(args.schema.read_text())
    jsonschema.Draft7Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(bom)
    args.output.write_text(json.dumps(bom, indent=2) + "\n")
    print(json.dumps({"status": "passed", "components": len(components)}))


if __name__ == "__main__":
    main()
