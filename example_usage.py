#!/usr/bin/env python3
"""
Example usage script for MedNER-DE Service.

This script demonstrates how to use the MedNER-DE Service for medical entity extraction.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any


async def test_single_extraction():
    """Test single text extraction."""
    print("🔍 Testing single text extraction...")

    sample_text = """
    Der Patient, ein 65-jähriger Mann, leidet an Diabetes mellitus Typ 2.
    Er nimmt täglich Metformin 500mg und Insulin. 
    Zusätzlich hat er eine Hypertonie und nimmt Lisinopril.
    """

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/extract", json={"text": sample_text}
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Found {len(result['entities'])} entities:")
                for entity in result["entities"]:
                    icd_info = (
                        f" (ICD: {entity['icd_code']})" if entity["icd_code"] else ""
                    )
                    category_info = (
                        f" [{entity['category']}]" if entity["category"] else ""
                    )
                    print(
                        f"  - {entity['text']} ({entity['label']}){category_info}{icd_info} - Confidence: {entity['confidence']:.2f}"
                    )
                print(f"⏱️  Processing time: {result['processing_time']:.3f}s")
            else:
                print(f"❌ Error: {response.status}")


async def test_extraction_with_stats():
    """Test extraction with detailed statistics."""
    print("\n📊 Testing extraction with statistics...")

    sample_text = """
    Patient mit LWS-Syndrom, Gangschulung empfohlen. 
    Paracetamol 500 mg 2× täglich. Krankengymnastik am Gerät (KGG).
    """

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/extract_with_stats", json={"text": sample_text}
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"✅ Found {len(result['entities'])} entities:")
                print("\n📋 Display format:")
                print(result["display_text"])
                print(f"\n📊 Statistics:")
                stats = result["statistics"]
                print(f"  - Total entities: {stats['total_entities']}")
                print(f"  - With ICD codes: {stats['with_icd_codes']}")
                print(f"  - Categories: {stats['by_category']}")
                print(f"  - Labels: {stats['by_label']}")
                print(f"  - Confidence: {stats['confidence_distribution']}")
            else:
                print(f"❌ Error: {response.status}")


async def test_batch_extraction():
    """Test batch text extraction."""
    print("\n📦 Testing batch extraction...")

    sample_texts = [
        "Der Patient hat Diabetes und nimmt Metformin.",
        "Die Patientin leidet an Hypertonie und nimmt Lisinopril.",
        "Der Arzt verschreibt Aspirin gegen die Kopfschmerzen.",
    ]

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/extract_batch", json={"texts": sample_texts}
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(
                    f"✅ Processed {result['batch_size']} texts in {result['total_processing_time']:.3f}s"
                )

                for i, extraction in enumerate(result["results"]):
                    print(
                        f"\n  Text {i+1}: Found {len(extraction['entities'])} entities"
                    )
                    for entity in extraction["entities"]:
                        print(f"    - {entity['text']} ({entity['label']})")
            else:
                print(f"❌ Error: {response.status}")


async def test_health_check():
    """Test service health check."""
    print("\n🏥 Testing health check...")

    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/health") as response:
            if response.status == 200:
                health = await response.json()
                print(f"✅ Service status: {health['status']}")
                print(f"📊 Models loaded: {health['models']}")
                print(f"📈 Total extractions: {health['stats']['total_extractions']}")
                print(
                    f"⏱️  Average processing time: {health['stats']['avg_processing_time']:.3f}s"
                )
            else:
                print(f"❌ Health check failed: {response.status}")


async def test_stats():
    """Test service statistics."""
    print("\n📊 Testing service statistics...")

    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/stats") as response:
            if response.status == 200:
                stats = await response.json()
                print("✅ Service statistics:")
                print(
                    f"  - Total extractions: {stats['extraction_stats']['total_extractions']}"
                )
                print(
                    f"  - Total entities: {stats['extraction_stats']['total_entities']}"
                )
                print(
                    f"  - Average processing time: {stats['extraction_stats']['avg_processing_time']:.3f}s"
                )
                print(f"  - Model status: {stats['model_status']}")
            else:
                print(f"❌ Stats request failed: {response.status}")


async def main():
    """Main example function."""
    print("🚀 MedNER-DE Service Example Usage")
    print("=" * 50)

    try:
        # Test health first
        await test_health_check()

        # Test single extraction
        await test_single_extraction()

        # Test extraction with statistics
        await test_extraction_with_stats()

        # Test batch extraction
        await test_batch_extraction()

        # Test statistics
        await test_stats()

        print("\n✅ All tests completed successfully!")

    except aiohttp.ClientConnectorError:
        print("❌ Could not connect to MedNER-DE Service.")
        print("   Make sure the service is running on http://localhost:8000")
        print("   Start the service with: python -m api.app")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
