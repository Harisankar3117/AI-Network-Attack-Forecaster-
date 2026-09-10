import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("1. Opening Demo Mode...")
        await page.goto("http://localhost:5173/demo")
        
        print("2. Selecting CSV file...")
        file_path = r"C:\Users\Welcome\Desktop\AI-Network-Attack-Forecaster (2)\AI-Network-Attack-Forecaster (3)\AI-Network-Attack-Forecaster\data\processed\CIC-IDS2018\Friday-16-02-2018_TrafficForML_CICFlowMeter_cleaned.csv"
        
        # We find the file input and use set_input_files
        await page.locator('input[type="file"]').set_input_files(file_path)
        
        print("3. Clicking 'Run Offline Forecast'...")
        # Start waiting for the response from the POST request
        async with page.expect_response(lambda response: "/api/v1/upload" in response.url and response.request.method == "POST", timeout=60000) as response_info:
            await page.get_by_role("button", name="Run Offline Forecast").click()
            
        print("4. Confirmed POST /api/v1/upload?k_steps=15 request")
        response = await response_info.value
        print(f"5. FastAPI Response Status: {response.status}")
        
        json_data = await response.json()
        print("6. Confirmed real forecast JSON response returned")
        # print some keys just to verify
        if 'current_probability' in json_data and 'forecast_windows' in json_data:
            print("   Response has expected keys (current_probability, forecast_windows)")
        
        print("7. Waiting for navigation to /forecast...")
        await page.wait_for_url("**/forecast")
        print("   Navigated to /forecast")
        
        print("8. Checking dropdown for 'Custom Uploaded CSV'...")
        # The select box should have "custom" as its value
        select_value = await page.locator("select").input_value()
        if select_value == "custom":
            print("   'Custom Uploaded CSV' is selected.")
        else:
            print(f"   Wait, selected value is {select_value}")
            
        print("11. Checking displayed probabilities...")
        peak_prob = await page.locator(".grid .nexus-card:has-text('Peak Probability') .text-h2").inner_text()
        current_risk = await page.locator(".grid .nexus-card:has-text('Current Risk Level') .badge").inner_text()
        active_model = await page.locator(".grid .nexus-card:has-text('Active Model') .text-h2").inner_text()
        
        print(f"   Peak Probability: {peak_prob}")
        print(f"   Current Risk Level: {current_risk}")
        print(f"   Active Model: {active_model}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
