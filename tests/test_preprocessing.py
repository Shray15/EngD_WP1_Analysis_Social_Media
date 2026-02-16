"""
Tests for text preprocessing utilities.

This module tests the text preprocessing functions used throughout
the social media analysis project.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the preprocessing function
from intent_utils.intent_train_test_preprocess import preprocess


class TestTextPreprocessing:
    """Test cases for text preprocessing functions."""

    def test_basic_preprocessing(self):
        """Test basic text preprocessing functionality."""
        text = "Hello world! How are you?"
        result = preprocess(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == text  # Should be unchanged for clean text

    def test_html_entity_unescaping(self):
        """Test HTML entity unescaping."""
        text = "&amp; &lt; &gt; &quot; &#39;"
        result = preprocess(text)
        
        assert "&amp;" not in result
        assert "&lt;" not in result
        assert "&gt;" not in result
        assert "&" in result or "<" in result or ">" in result

    def test_html_tag_removal(self):
        """Test HTML tag removal."""
        text = "<p>This is <b>bold</b> text</p>"
        result = preprocess(text)
        
        assert "<p>" not in result
        assert "<b>" not in result
        assert "</b>" not in result
        assert "</p>" not in result
        assert "This is" in result
        assert "bold" in result
        assert "text" in result

    def test_url_standardization(self):
        """Test URL standardization."""
        test_cases = [
            "Visit https://example.com for info",
            "Check out www.example.com", 
            "Go to http://subdomain.example.com/path"
        ]
        
        for text in test_cases:
            result = preprocess(text)
            assert "<URL>" in result
            assert "http" not in result
            assert "www." not in result or "<URL>" in result

    def test_email_standardization(self):
        """Test email standardization."""
        text = "Contact us at test@example.com or support@company.org"
        result = preprocess(text)
        
        assert "<EMAIL>" in result
        assert "@example.com" not in result
        assert "@company.org" not in result

    def test_user_mention_standardization(self):
        """Test user mention standardization."""
        text = "Thanks @user123 and @another_user for help!"
        result = preprocess(text)
        
        assert "<USER>" in result
        assert "@user123" not in result
        assert "@another_user" not in result

    def test_number_standardization(self):
        """Test number standardization."""
        test_cases = [
            "The price is $29.99",
            "Count: 123 items",
            "Ratio is 3.14159",
            "European format: 1.234,56"
        ]
        
        for text in test_cases:
            result = preprocess(text)
            assert "<NUMBER>" in result

    def test_whitespace_normalization(self):
        """Test whitespace normalization.""" 
        text = "Multiple    spaces   and\ttabs\nand   newlines"
        result = preprocess(text)
        
        # Should not have multiple consecutive spaces
        assert "    " not in result
        assert "   " not in result
        assert "\t" not in result

    def test_punctuation_normalization(self):
        """Test punctuation spacing and runs."""
        text = "What ?!?!   And this , that ;  colon :"
        result = preprocess(text)
        
        # No space before punctuation
        assert " ?" not in result
        assert " !" not in result
        assert " ," not in result
        assert " ;" not in result
        assert " :" not in result

    def test_empty_string(self):
        """Test handling of empty strings."""
        result = preprocess("")
        assert result == ""

    def test_none_input(self):
        """Test handling of None input."""
        result = preprocess(None)
        assert result == ""

    def test_unicode_normalization(self):
        """Test Unicode normalization (NFKC)."""
        # Test with some Unicode characters that should be normalized
        text = "café naïve résumé"  # Contains accented characters
        result = preprocess(text)
        
        # Should preserve the accented characters (NFKC normalization)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_case_preservation(self):
        """Test that case is preserved."""
        text = "This is a MiXeD CaSe TeXt"
        result = preprocess(text)
        
        # Should preserve original case
        assert "MiXeD" in result
        assert "CaSe" in result

    def test_emoji_preservation(self):
        """Test that emojis are preserved."""
        text = "Great job! 😀 🎉 ❤️"
        result = preprocess(text)
        
        # Emojis should be preserved
        assert len(result) > len("Great job!")

    def test_complex_example(self):
        """Test preprocessing with complex real-world example."""
        text = """
        <p>Check out https://example.com/article @user123!</p>
        Contact: support@company.com for help.
        Price: $199.99 (was $299.99) 
        Rating: 4.5/5.0 ⭐⭐⭐⭐⭐
        &quot;Amazing product!&quot; says customer.
        """
        
        result = preprocess(text)
        
        # Check that all transformations were applied
        assert "<URL>" in result
        assert "<EMAIL>" in result  
        assert "<USER>" in result
        assert "<NUMBER>" in result
        assert "<p>" not in result
        assert "</p>" not in result
        assert "&quot;" not in result
        assert "@user123" not in result
        assert "support@company.com" not in result
        assert "https://example.com" not in result
        
        # Check that content is preserved
        assert "Check out" in result
        assert "Amazing product" in result
        assert "customer" in result

    def test_dutch_text_handling(self):
        """Test handling of Dutch language text."""
        dutch_texts = [
            "Hoe gaat het met je?",
            "Dank je wel voor je hulp!",
            "Ik ben zeer tevreden met dit product.",
            "Wat kost de verzending naar Nederland?"
        ]
        
        for text in dutch_texts:
            result = preprocess(text)
            assert isinstance(result, str)
            assert len(result) > 0
            # Dutch characters should be preserved
            assert "ë" in result if "ë" in text else True
            assert "ï" in result if "ï" in text else True

    @pytest.mark.parametrize("input_text,expected_contains", [
        ("Visit https://test.com", "<URL>"),
        ("Email me@test.com", "<EMAIL>"),
        ("Hey @username", "<USER>"), 
        ("Count: 42", "<NUMBER>"),
        ("<b>Bold</b>", "Bold"),
        ("&amp;", "&")
    ])
    def test_parametrized_preprocessing(self, input_text, expected_contains):
        """Parametrized test for various preprocessing scenarios."""
        result = preprocess(input_text)
        assert expected_contains in result


if __name__ == "__main__":
    pytest.main([__file__])