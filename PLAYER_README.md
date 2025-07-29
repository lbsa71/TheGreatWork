# Static HTML Dialogue Tree Player

A standalone HTML player for dialogue trees that can be embedded into any website without requiring a server.

## Features

- **Completely Static**: No server required, runs entirely in the browser
- **Embeddable**: Can be embedded into any website via iframe or direct inclusion
- **JSON-driven**: Loads dialogue trees from JSON files
- **Interactive**: Navigate through dialogue trees with choice-based progression
- **Parameter Tracking**: Game parameters are updated based on player choices
- **History**: View the path taken through the dialogue tree
- **Responsive**: Works on desktop and mobile devices
- **Error Handling**: Graceful handling of missing or malformed JSON files

## Usage

### Basic Usage

1. Place `player.html` in your web directory alongside your dialogue tree JSON file
2. Open `player.html` in a web browser
3. By default, it will load `tree.json` from the same directory

### Custom JSON File

Use the `json` query parameter to specify a different JSON file:

```
player.html?json=my-story.json
player.html?json=stories/adventure.json
```

### Embedding in a Website

#### Option 1: iframe (Recommended)
```html
<iframe src="player.html?json=my-story.json" 
        width="800" 
        height="600" 
        style="border: none; border-radius: 8px;">
</iframe>
```

#### Option 2: Direct Integration
Since the player is a single HTML file with inline CSS and JavaScript, you can also copy the relevant parts into your existing webpage.

## JSON Format

The player expects a JSON file with the following structure:

```json
{
  "nodes": {
    "start": {
      "situation": "You find yourself at a crossroads...",
      "choices": [
        {
          "text": "Go left",
          "next": "left_path",
          "effects": {
            "courage": 5,
            "wisdom": -2
          }
        },
        {
          "text": "Go right", 
          "next": "right_path",
          "effects": {
            "courage": -3,
            "stealth": 10
          }
        }
      ]
    },
    "left_path": {
      "situation": "You take the left path...",
      "choices": [...]
    },
    "right_path": {
      "situation": "You take the right path...",
      "choices": []
    }
  },
  "params": {
    "courage": 50,
    "wisdom": 25,
    "stealth": 30
  },
  "rules": {
    "language": "English",
    "tone": "adventurous",
    "voice": "second person",
    "style": "fantasy adventure"
  },
  "scene": {
    "setting": "A mystical forest",
    "atmosphere": "mysterious"
  }
}
```

### Required Fields

- `nodes`: Object containing all dialogue nodes
  - Each node must have a `situation` (string) describing the current state
  - Each node can have `choices` (array) for player options
  - Nodes with empty `choices` arrays are treated as ending nodes
- `params`: Object containing initial game parameter values

### Optional Fields

- `rules`: Metadata about the story style (not used by player but preserved)
- `scene`: Metadata about the story setting (not used by player but preserved)

### Choice Format

Each choice in a node's `choices` array can have:
- `text` (required): The text displayed to the player
- `next` (required): The ID of the node to navigate to
- `effects` (optional): Object with parameter changes (e.g., `{"courage": 5, "wisdom": -2}`)

## Example Files

The repository includes:
- `tree.json`: Complex sci-fi story with cosmic horror themes
- `test-tree.json`: Simpler fantasy adventure for testing

## Browser Compatibility

The player uses modern JavaScript features and should work in:
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

For older browsers, consider using a polyfill service.

## Development

The player is contained entirely in `player.html` with inline CSS and JavaScript for maximum portability. To modify:

1. Edit the CSS in the `<style>` section for visual changes
2. Edit the JavaScript in the `<script>` section for functionality changes
3. Test with your JSON files

## License

This static player is part of TheGreatWork project and follows the same license terms.