import requests
import re

# 🔴 Update with your ngrok HTTPS URL for the test server (port 3001)
NGROK_URL = "https://subattenuate-joanne-vacuous.ngrok-free.dev"  # Local test, or replace with ngrok URL


def clean_product_name(product_title):
    """Clean product name by removing site domains and unwanted text"""
    if not product_title:
        return product_title

    remove_patterns = [
        r"\s*[-|]\s*(amazon|aliexpress|ebay|walmart|carrefour|jumia|souq|noon|alibaba|pinterest|etsy)",
        r"\s*[-|]\s*[\w\-]+\.(com|net|org|co|fr|tn|uk|us)",
        r"^\s*Best.*?:",
        r"\s*(Online|Shop|Store|Buy|Sale|Deal|Offer)\s*$",
    ]

    cleaned = product_title
    for pattern in remove_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else product_title


def upload_image_and_segment(image_path):
    """Upload image and get segmentation results"""
    endpoint = f"{NGROK_URL}/analyze"

    try:
        with open(image_path, "rb") as img:
            files = {"image": img}
            response = requests.post(endpoint, files=files)

        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None

        data = response.json()

        # Extract from segmentation result
        segmentation = data.get("segmentation", {})
        products = segmentation.get("products", [])

        print(f"\n✅ Segmented {len(products)} products:\n")
        for i, product in enumerate(products, 1):
            label = product.get("label", "product")
            confidence = product.get("confidence", 0)
            crop_path = product.get("crop_path", "N/A")

            print(f"  {i}. {label}")
            print(f"     Confidence: {confidence:.2%}")
            print(f"     Crop: {crop_path}\n")

        return products

    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def scrape_product_ingredients(selected_product):
    """Call /scrape-ingredients endpoint with selected segmentation product data"""
    endpoint = f"{NGROK_URL}/scrape-ingredients"

    try:
        detected_label = selected_product.get("label", "product")
        payload = {
            "crop_path": selected_product.get("crop_path"),
            "detected_label": detected_label,
        }

        print(f"\n🔍 Running Lens -> OFF -> EDC for detected product: {detected_label}...")
        response = requests.post(endpoint, json=payload)

        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None

        data = response.json()
        return data

    except Exception as e:
        print(f"❌ Error scraping: {str(e)}")
        return None


def display_ingredients_with_edc(ingredients_data):
    """Display extracted ingredients with EDC enrichment"""
    if not ingredients_data:
        return

    source = ingredients_data.get("source", "Unknown")
    product_name = ingredients_data.get("product_name", "Unknown")
    ingredients = ingredients_data.get("ingredients", "Not found")
    resolved_name = ingredients_data.get("resolved_product_name", product_name)
    lens_error = ingredients_data.get("lens_error")
    lens_matches = ingredients_data.get("lens_matches", [])
    public_image_url = ingredients_data.get("public_image_url")

    print(f"\n✅ Ingredients for: {product_name}")
    print(f"   Resolved Name: {resolved_name}")
    print(f"   Source: {source}")
    if public_image_url:
        print(f"   Lens Image URL: {public_image_url}")
    if lens_matches:
        best_title = lens_matches[0].get("title", "Unknown")
        print(f"   Lens Top Match: {best_title}")
    if lens_error:
        print(f"   Lens Warning: {lens_error}")
    print(f"   {ingredients}")

    # Display TEDX/EDC information if available
    edc_list = ingredients_data.get("edc_list", [])
    if edc_list:
        print(f"\n⚠️  EDC (Endocrine Disruptor) Matches: {len(edc_list)}")
        for hit in edc_list:
            additive = hit.get("additive")
            chemical = hit.get("chemical_name")
            score = hit.get("match_score", 0)
            print(f"\n   • {additive} ({chemical})")
            print(f"     Match Score: {score:.1f}%")

            # Health effects per list
            health_effects = hit.get("health_effects", {})
            if any(health_effects.values()):
                print(f"     Health Effects:")
                for list_key, effect in health_effects.items():
                    if effect:
                        print(f"       - {list_key}: {effect}")

            # Environmental effects per list
            env_effects = hit.get("environmental_effects", {})
            if any(env_effects.values()):
                print(f"     Environmental Effects:")
                for list_key, effect in env_effects.items():
                    if effect:
                        print(f"       - {list_key}: {effect}")

            # PubChem enrichment
            pubchem = hit.get("pubchem")
            if pubchem and pubchem.get("cid"):
                print(f"     PubChem Details:")
                print(f"       - CID: {pubchem.get('cid')}")
                structure = pubchem.get("structure", {})
                if structure.get("molecular_formula"):
                    print(f"       - Formula: {structure.get('molecular_formula')}")
                if structure.get("molecular_weight"):
                    print(f"       - Weight: {structure.get('molecular_weight')}")

    # Display any errors loading reference data
    errors = {
        "tedx_error": ingredients_data.get("tedx_error"),
        "list1_error": ingredients_data.get("list1_error"),
        "list2_error": ingredients_data.get("list2_error"),
        "list3_error": ingredients_data.get("list3_error"),
    }
    for error_key, error_val in errors.items():
        if error_val:
            print(f"\n⚠️  {error_key}: {error_val}")

    print(f"\n✅ Saved at: {ingredients_data.get('scanned_at', 'N/A')}")


def main():
    """Main flow: segment image, then check ingredients + EDC for each product"""
    image_path = "brownies.jpg"

    # Step 1: Upload image and segment
    products = upload_image_and_segment(image_path)
    if not products:
        return

    # Step 2: Prompt user to select a product
    while True:
        try:
            choice = (
                input(
                    "\n📌 Enter product number to check ingredients (or 'q' to quit): "
                )
                .strip()
            )
            if choice.lower() == "q":
                print("👋 Goodbye!")
                break

            product_idx = int(choice) - 1
            if product_idx < 0 or product_idx >= len(products):
                print("❌ Invalid choice. Try again.")
                continue

            selected_product = products[product_idx]

            # Step 3: Scrape ingredients + EDC
            ingredients_data = scrape_product_ingredients(selected_product)

            # Step 4: Display results with EDC enrichment
            if ingredients_data:
                display_ingredients_with_edc(ingredients_data)
                print("✅ Data saved to ingredients.csv")

            # Ask if they want to check another
            again = input("\n🔄 Check another product? (y/n): ").strip().lower()
            if again != "y":
                print("👋 Goodbye!")
                break

        except ValueError:
            print("❌ Please enter a valid number.")
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
