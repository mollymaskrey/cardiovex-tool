"""
app.py

Cardiovex Field Rep Trainer — main Dash application.
Pharma field force Socratic training simulator.

Start command (local):  python app.py
Start command (Render): gunicorn app:server

Architecture:
  app-state  — screen, mode, persona, rep_name (no history)
  chat-store — conversation history only
  mode-store — sim/oracle (read by begin-btn)

Keeping history in its own store prevents render_chat from
firing when persona/screen state changes.
"""

import os
import anthropic
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

from personas import get_persona, PERSONAS
from rag import CardiovexRAG

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

COLORS = {
    "bg":       "#000000",
    "panel":    "#0a0f0a",
    "border":   "#0d2b0d",
    "cyan":     "#00ffcc",
    "cyan_dim": "#00aa88",
    "green":    "#00ff88",
    "red":      "#ff3355",
    "text":     "#ccffee",
    "text_dim": "#557766",
    "cardiologist_visit":      "#2ecc71",
    "prior_auth":              "#3498db",
    "adverse_event":           "#e74c3c",
    "formulary_committee":     "#f39c12",
    "patient_education":       "#9b59b6",
    "clinical_trial":          "#1abc9c",
}

FONT_MONO    = "'Share Tech Mono', 'Courier New', monospace"
FONT_DISPLAY = "'Orbitron', 'Share Tech Mono', monospace"

EXTERNAL_STYLESHEETS = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap",
]

app = dash.Dash(
    __name__,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    suppress_callback_exceptions=True,
    title="Cardiovex Field Rep Trainer",
)
server = app.server

# Initialize RAG system
rag_system = CardiovexRAG(db_path="data/chroma_db")

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def panel_style(extra=None):
    base = {
        "backgroundColor": COLORS["panel"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "2px",
        "padding": "24px",
    }
    if extra:
        base.update(extra)
    return base

def btn_primary(active=False, color=None):
    c = color or COLORS["cyan"]
    return {
        "backgroundColor": c if active else "transparent",
        "color": COLORS["bg"] if active else c,
        "border": f"1px solid {c}",
        "borderRadius": "2px",
        "fontFamily": FONT_DISPLAY,
        "fontSize": "15px",
        "letterSpacing": "3px",
        "padding": "12px 24px",
        "cursor": "pointer",
        "width": "100%",
        "textTransform": "uppercase",
    }

def lbl():
    return {
        "color": COLORS["text_dim"],
        "fontSize": "15px",
        "letterSpacing": "3px",
        "marginBottom": "8px",
        "fontFamily": FONT_DISPLAY,
    }

GRID_STYLE = {
    "position": "fixed",
    "top": 0, "left": 0,
    "width": "100%", "height": "100%",
    "backgroundImage": (
        f"linear-gradient({COLORS['border']} 1px, transparent 1px),"
        f"linear-gradient(90deg, {COLORS['border']} 1px, transparent 1px)"
    ),
    "backgroundSize": "40px 40px",
    "pointerEvents": "none",
    "zIndex": 0,
    "opacity": 0.4,
}

# ---------------------------------------------------------------------------
# Claude API helpers
# ---------------------------------------------------------------------------

import re

def strip_markdown(text: str) -> str:
    """Convert markdown to clean plain text for display, but preserve stage direction markers."""
    text = re.sub(r'\n---+\n', '\n\n', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # DON'T strip single asterisks - those are stage directions
    # Only strip bold (**text**) and bold-italic (***text***)
    text = re.sub(r'\*{2,3}([^*]+)\*{2,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
    text = re.sub(r'^\s*[-]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'✅\s*', '✓ ', text)
    text = re.sub(r'❌\s*', '✗ ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def call_claude(system_prompt, messages):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    return strip_markdown(response.content[0].text)

def get_opening_line(mode, persona_key, rep_name):
    if mode == "oracle":
        return (
            f"Welcome {rep_name}. I'm your Cardiovex expert — think of me as your MSL "
            f"on speed dial. Ask me anything: objection handling, trial data, label language, "
            f"market access. What do you want to work through?"
        )
    
    # For simulation mode, get context and use full persona prompt
    try:
        chunks = rag_system.retrieve("cardiovex secondary prevention MACCE SHIELD", n_results=4)
        context = rag_system.format_context(chunks)
    except Exception:
        context = "(Context retrieval unavailable.)"
    
    system_prompt = get_persona(persona_key, context_block=context)
    
    try:
        response = call_claude(system_prompt, [{"role": "user", "content": "BEGIN SESSION"}])
        return response
    except Exception as e:
        return f"[Error: {str(e)}]"

# ---------------------------------------------------------------------------
# Header / Footer
# ---------------------------------------------------------------------------

def header_bar():
    return html.Div([
        html.Div([
            html.Span("NOVELLIA", style={"color": COLORS["cyan"], "fontFamily": FONT_DISPLAY, "fontSize": "16px", "letterSpacing": "4px"}),
            html.Span(" · ", style={"color": COLORS["text_dim"], "margin": "0 8px"}),
            html.Span("CARDIOVEX", style={"color": COLORS["cyan"], "fontFamily": FONT_DISPLAY, "fontSize": "16px", "letterSpacing": "4px"}),
            html.Span(" · ", style={"color": COLORS["text_dim"], "margin": "0 8px"}),
            html.Span("FIELD REP", style={"color": COLORS["cyan"], "fontFamily": FONT_DISPLAY, "fontSize": "16px", "letterSpacing": "4px"}),
        ], style={"textAlign": "center", "padding": "16px 0 8px 0"}),
        html.H1("CARDIOVEX FIELD REP TRAINER", style={
            "fontFamily": FONT_DISPLAY,
            "fontSize": "clamp(24px, 5vw, 48px)",
            "fontWeight": "900",
            "color": COLORS["cyan"],
            "textAlign": "center",
            "letterSpacing": "8px",
            "margin": "8px 0",
            "textShadow": f"0 0 30px {COLORS['cyan']}44",
        }),
        html.Div("PHARMACEUTICAL FIELD FORCE TRAINING SYSTEM", style={
            "textAlign": "center",
            "color": COLORS["cyan_dim"],
            "fontFamily": FONT_DISPLAY,
            "fontSize": "14px",
            "letterSpacing": "5px",
            "paddingBottom": "8px",
        }),
        # SYNTHETIC DATA MARKER
        html.Div([
            html.Span("⚠ ", style={"color": COLORS["red"]}),
            html.Span("SIMULATED DATA ONLY — CARDIOVEX IS A FICTIONAL PRODUCT", style={
                "color": COLORS["red"],
                "fontFamily": FONT_DISPLAY,
                "fontSize": "12px",
                "letterSpacing": "2px",
            }),
            html.Span(" ⚠", style={"color": COLORS["red"]}),
        ], style={"textAlign": "center", "paddingBottom": "16px"}),
    ], style={"borderBottom": f"1px solid {COLORS['border']}", "position": "relative", "zIndex": 10})

def footer_bar():
    return html.Div([
        html.Span("● SYSTEM ONLINE", style={"color": COLORS["green"]}),
        html.Span("MOLLY MASKREY AI // NOVELLIA CARDIOVEX // SIMULATED DATA", style={"color": COLORS["text_dim"]}),
    ], style={
        "display": "flex",
        "justifyContent": "space-between",
        "padding": "12px 24px",
        "fontSize": "14px",
        "letterSpacing": "2px",
        "fontFamily": FONT_DISPLAY,
        "borderTop": f"1px solid {COLORS['border']}",
    })

# ---------------------------------------------------------------------------
# Splash screen
# ---------------------------------------------------------------------------

def splash_screen():
    return html.Div([
        html.Div(style=GRID_STYLE),
        header_bar(),
        html.Div([
            html.Div([
                html.Div("SELECT TRAINING MODE", style={**lbl(), "textAlign": "center", "marginBottom": "24px"}),
                html.Button("SIMULATION MODE", id="mode-sim-btn", n_clicks=0, style=btn_primary()),
                html.Div("Roleplay scenarios with AI personas", style={
                    "color": COLORS["text_dim"],
                    "fontSize": "13px",
                    "textAlign": "center",
                    "marginTop": "8px",
                    "marginBottom": "24px",
                }),
                html.Button("ORACLE MODE", id="mode-oracle-btn", n_clicks=0, style=btn_primary()),
                html.Div("Expert Q&A on Cardiovex data", style={
                    "color": COLORS["text_dim"],
                    "fontSize": "13px",
                    "textAlign": "center",
                    "marginTop": "8px",
                }),
            ], style=panel_style({"maxWidth": "600px", "margin": "80px auto"})),
        ], style={"position": "relative", "zIndex": 10}),
        footer_bar(),
    ], style={"backgroundColor": COLORS["bg"], "color": COLORS["text"], "minHeight": "100vh", "fontFamily": FONT_MONO})

# ---------------------------------------------------------------------------
# Name entry screen
# ---------------------------------------------------------------------------

def name_screen(mode):
    mode_label = "SIMULATION" if mode == "sim" else "ORACLE"
    return html.Div([
        html.Div(style=GRID_STYLE),
        header_bar(),
        html.Div([
            html.Div([
                html.Div(f"{mode_label} MODE", style={**lbl(), "textAlign": "center", "color": COLORS["cyan"], "marginBottom": "24px"}),
                html.Div("ENTER YOUR NAME", style={**lbl(), "textAlign": "center"}),
                dcc.Input(
                    id="rep-name-input",
                    type="text",
                    placeholder="Rep Name",
                    style={
                        "width": "100%",
                        "padding": "16px",
                        "backgroundColor": COLORS["bg"],
                        "color": COLORS["cyan"],
                        "border": f"1px solid {COLORS['cyan']}",
                        "borderRadius": "2px",
                        "fontFamily": FONT_MONO,
                        "fontSize": "16px",
                        "marginBottom": "24px",
                    },
                ),
                html.Button("BEGIN SESSION", id="begin-btn", n_clicks=0, style=btn_primary(active=True)),
            ], style=panel_style({"maxWidth": "500px", "margin": "120px auto"})),
        ], style={"position": "relative", "zIndex": 10}),
        footer_bar(),
        dcc.Store(id="mode-store", data=mode),
    ], style={"backgroundColor": COLORS["bg"], "color": COLORS["text"], "minHeight": "100vh", "fontFamily": FONT_MONO})

# ---------------------------------------------------------------------------
# Persona selector (for simulation mode only)
# ---------------------------------------------------------------------------

def persona_selector_bar():
    """Create persona selector buttons for Patricia, Dr. Chen, Margaret."""
    buttons = []
    for key in ["patricia", "dr_chen", "margaret"]:
        meta = PERSONAS[key]
        color = COLORS.get(key.replace("_", ""), COLORS["cyan"])
        buttons.append(
            html.Div([
                html.Button(
                    meta["name"],
                    id=f"persona-{key}-btn",
                    n_clicks=0,
                    style=btn_primary(color=color)
                ),
            ], style={"marginBottom": "12px"})
        )
    
    return html.Div([
        html.Div("SELECT SCENARIO", style={**lbl(), "marginBottom": "16px"}),
        html.Div(buttons),
        html.Div(id="persona-description", style={
            "marginTop": "16px",
            "padding": "12px",
            "backgroundColor": COLORS["bg"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "2px",
            "fontSize": "13px",
            "color": COLORS["text_dim"],
        }),
    ], style=panel_style({"marginBottom": "24px"}))

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

def render_message(role, content, persona_name="Expert"):
    """Render a chat message with stage directions styled differently."""
    is_user = (role == "user")
    
    # Parse out stage directions - ONLY text between *asterisks* or [brackets]
    import re
    
    # Split content into parts: regular text and stage directions
    parts = []
    current_pos = 0
    
    # Find stage directions: *action* or [action] only
    pattern = r'(\*[^*]+\*|\[[^\]]+\])'
    
    for match in re.finditer(pattern, content):
        # Add text before the stage direction
        if match.start() > current_pos:
            parts.append({
                "text": content[current_pos:match.start()],
                "is_stage_direction": False
            })
        
        # Add the stage direction (keep the markers for now, we'll style them)
        parts.append({
            "text": match.group(0),
            "is_stage_direction": True
        })
        
        current_pos = match.end()
    
    # Add remaining text
    if current_pos < len(content):
        parts.append({
            "text": content[current_pos:],
            "is_stage_direction": False
        })
    
    # If no stage directions found, just use the whole content
    if not parts:
        parts = [{"text": content, "is_stage_direction": False}]
    
    # Build the message content with styled parts
    message_content = []
    for part in parts:
        if part["is_stage_direction"]:
            message_content.append(
                html.Span(
                    part['text'],
                    style={
                        "color": COLORS["text_dim"],
                        "fontStyle": "italic",
                        "fontSize": "14px",
                    }
                )
            )
        else:
            message_content.append(part["text"])
    
    return html.Div([
        html.Div(
            "YOU" if is_user else persona_name.upper(),
            style={
                "color": COLORS["cyan"] if is_user else COLORS["green"],
                "fontSize": "13px",
                "letterSpacing": "3px",
                "marginBottom": "8px",
                "fontFamily": FONT_DISPLAY,
            }
        ),
        html.Div(
            message_content,
            style={
                "color": COLORS["text"],
                "fontSize": "15px",
                "lineHeight": "1.6",
                "whiteSpace": "pre-wrap",
            }
        ),
    ], style={
        "marginBottom": "32px",
        "paddingBottom": "24px",
        "borderBottom": f"1px solid {COLORS['border']}",
    })

def chat_screen():
    return html.Div([
        html.Div(style=GRID_STYLE),
        header_bar(),
        html.Div([
            # Left sidebar - persona selector (FIXED, no scroll)
            html.Div(
                id="persona-selector-container",
                style={
                    "width": "280px", 
                    "marginRight": "24px",
                    "flexShrink": "0",
                }
            ),
            # Main chat area - FIXED 1000px height
            html.Div([
                # Status bar
                html.Div(
                    id="session-status-bar",
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "padding": "16px 0",
                        "fontSize": "13px",
                        "letterSpacing": "2px",
                        "fontFamily": FONT_DISPLAY,
                        "borderBottom": f"1px solid {COLORS['border']}",
                        "marginBottom": "24px",
                    }
                ),
                # Chat messages - SCROLLING within fixed height
                html.Div(
                    id="chat-messages",
                    style={
                        "height": "700px",
                        "overflowY": "auto",
                        "marginBottom": "24px",
                        "paddingRight": "12px",
                    }
                ),
                # Input area - FIXED at bottom
                html.Div([
                    dcc.Textarea(
                        id="user-input",
                        placeholder="Type your response...",
                        style={
                            "width": "100%",
                            "height": "120px",
                            "padding": "16px",
                            "backgroundColor": COLORS["bg"],
                            "color": COLORS["cyan"],
                            "border": f"1px solid {COLORS['cyan']}",
                            "borderRadius": "2px",
                            "fontFamily": FONT_MONO,
                            "fontSize": "15px",
                            "resize": "none",
                            "marginBottom": "16px",
                        },
                    ),
                    html.Div([
                        html.Button("SEND", id="send-btn", n_clicks=0, style={**btn_primary(active=True), "width": "120px", "marginRight": "12px"}),
                        html.Button("NEW SESSION", id="new-session-btn", n_clicks=0, style={**btn_primary(), "width": "180px", "marginRight": "12px"}),
                        html.Button("EXPORT", id="export-btn", n_clicks=0, style={**btn_primary(), "width": "120px"}),
                    ], style={"display": "flex"}),
                ], style=panel_style()),
            ], style={"flex": "1", "height": "1000px", "display": "flex", "flexDirection": "column"}),
        ], style={
            "display": "flex",
            "padding": "24px",
            "position": "relative",
            "zIndex": 10,
        }),
        footer_bar(),
        dcc.Download(id="export-download"),
    ], style={"backgroundColor": COLORS["bg"], "color": COLORS["text"], "minHeight": "100vh", "fontFamily": FONT_MONO})

# ---------------------------------------------------------------------------
# Main layout with router
# ---------------------------------------------------------------------------

app.layout = html.Div([
    dcc.Store(id="app-state", data={"screen": "splash", "mode": "sim", "persona": "patricia", "rep_name": ""}),
    dcc.Store(id="chat-store", data=[]),
    html.Div(id="page-content"),
])

@app.callback(
    Output("page-content", "children"),
    Input("app-state", "data"),
)
def route_page(state):
    screen = state.get("screen", "splash")
    if screen == "splash":
        return splash_screen()
    elif screen == "name":
        return name_screen(state.get("mode", "sim"))
    elif screen == "chat":
        return chat_screen()
    return splash_screen()

# ---------------------------------------------------------------------------
# Mode selection — writes app-state only
# ---------------------------------------------------------------------------

@app.callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("mode-sim-btn", "n_clicks"),
    Input("mode-oracle-btn", "n_clicks"),
    prevent_initial_call=True,
)
def select_mode(sim_clicks, oracle_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    triggered = ctx.triggered[0]["prop_id"]
    mode = "oracle" if "oracle" in triggered else "sim"
    return {"screen": "name", "mode": mode, "persona": "patricia", "rep_name": ""}

# ---------------------------------------------------------------------------
# Begin session — writes both app-state AND chat-store
# ---------------------------------------------------------------------------

@app.callback(
    Output("app-state", "data", allow_duplicate=True),
    Output("chat-store", "data", allow_duplicate=True),
    Input("begin-btn", "n_clicks"),
    State("rep-name-input", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def begin_session(n_clicks, rep_name, state):
    if not n_clicks or not rep_name:
        return no_update, no_update
    
    mode = state.get("mode", "sim")
    persona = state.get("persona", "cardiologist_visit")
    
    new_state = {
        "screen": "chat",
        "mode": mode,
        "persona": persona,
        "rep_name": rep_name.strip(),
    }
    
    try:
        opening = get_opening_line(mode, persona, rep_name)
    except Exception as e:
        opening = f"[Error: {str(e)}]"
    
    new_history = [{"role": "assistant", "content": opening}]
    
    return new_state, new_history

# ---------------------------------------------------------------------------
# New session — resets both stores
# ---------------------------------------------------------------------------

@app.callback(
    Output("app-state", "data", allow_duplicate=True),
    Output("chat-store", "data", allow_duplicate=True),
    Input("new-session-btn", "n_clicks"),
    prevent_initial_call=True,
)
def new_session(n_clicks):
    if not n_clicks:
        return no_update, no_update
    return {
        "screen": "splash",
        "mode": "sim",
        "persona": "patricia",
        "rep_name": "",
    }, []

# ---------------------------------------------------------------------------
# Persona selection — writes app-state AND chat-store
# ---------------------------------------------------------------------------

@app.callback(
    Output("app-state", "data", allow_duplicate=True),
    Output("chat-store", "data", allow_duplicate=True),
    Output("persona-description", "children"),
    Output("persona-patricia-btn", "style"),
    Output("persona-dr_chen-btn", "style"),
    Output("persona-margaret-btn", "style"),
    Input("persona-patricia-btn", "n_clicks"),
    Input("persona-dr_chen-btn", "n_clicks"),
    Input("persona-margaret-btn", "n_clicks"),
    State("app-state", "data"),
    State("chat-store", "data"),
    prevent_initial_call=True,
)
def select_persona(p, c, m, state, current_history):
    ctx = callback_context
    if not ctx.triggered:
        return [no_update] * 6
    
    triggered = ctx.triggered[0]["prop_id"]
    
    # Find which persona was clicked
    if "patricia" in triggered:
        selected_key = "patricia"
    elif "dr_chen" in triggered:
        selected_key = "dr_chen"
    else:
        selected_key = "margaret"
    
    # Check if this is actually a change
    if state.get("persona") == selected_key and current_history:
        # Same persona already selected, don't regenerate
        return [no_update] * 6
    
    meta = PERSONAS[selected_key]
    rep_name = state.get("rep_name", "Rep")
    
    try:
        opening = get_opening_line("sim", selected_key, rep_name)
    except Exception as e:
        opening = f"[Error: {str(e)}]"
    
    # Build button styles
    def s(k):
        color = COLORS.get(k.replace("_", ""), COLORS["cyan"])
        return btn_primary(active=(k == selected_key), color=color)
    
    new_state = {**state, "persona": selected_key}
    new_history = [{"role": "assistant", "content": opening}]
    description = f"{meta['title']} · {meta['role']} · {meta['label']}"
    
    return new_state, new_history, description, s("patricia"), s("dr_chen"), s("margaret")

# ---------------------------------------------------------------------------
# Update chrome (persona selector + status bar) — driven by app-state only
# ---------------------------------------------------------------------------

@app.callback(
    Output("persona-selector-container", "children"),
    Output("session-status-bar", "children"),
    Input("app-state", "data"),
)
def update_chrome(state):
    if not state or state.get("screen") != "chat":
        return [], []
    
    mode = state.get("mode", "sim")
    persona_key = state.get("persona", "patricia")
    rep_name = state.get("rep_name", "")
    
    if mode == "oracle":
        selector = []
        status = [
            html.Span(f"// {rep_name.upper()}", style={"color": COLORS["cyan"]}),
            html.Span("MODE: ORACLE · EXPERT Q&A", style={"color": COLORS["text_dim"]}),
        ]
    else:
        selector = [persona_selector_bar()]
        meta = PERSONAS.get(persona_key, PERSONAS["patricia"])
        color = COLORS.get(persona_key.replace("_", ""), COLORS["cyan"])
        status = [
            html.Span(f"// {rep_name.upper()}", style={"color": COLORS["cyan"]}),
            html.Span(f"MODE: SIMULATION · {meta['name'].upper()} · {meta['label'].upper()}", style={"color": color}),
        ]
    
    return selector, status

# ---------------------------------------------------------------------------
# Render chat — driven by chat-store ONLY
# ---------------------------------------------------------------------------

@app.callback(
    Output("chat-messages", "children"),
    Input("chat-store", "data"),
    State("app-state", "data"),
)
def render_chat(history, state):
    if not history or not state or state.get("screen") != "chat":
        return []
    mode = state.get("mode", "sim")
    persona_key = state.get("persona", "patricia")
    persona_name = PERSONAS.get(persona_key, {}).get("name", "Expert") if mode == "sim" else "Expert"
    return [render_message(m["role"], m["content"], persona_name) for m in history]

# ---------------------------------------------------------------------------
# Send message — writes chat-store only, clears textarea
# ---------------------------------------------------------------------------

@app.callback(
    Output("chat-store", "data", allow_duplicate=True),
    Output("user-input", "value"),
    Input("send-btn", "n_clicks"),
    State("user-input", "value"),
    State("app-state", "data"),
    State("chat-store", "data"),
    prevent_initial_call=True,
)
def send_message(n_clicks, user_text, state, history):
    if not n_clicks or not user_text or not user_text.strip():
        return no_update, no_update
    
    user_text = user_text.strip()
    mode = state.get("mode", "sim")
    persona_key = state.get("persona", "patricia")
    rep_name = state.get("rep_name", "Rep")
    history = history or []
    
    # Retrieve context using RAG
    try:
        chunks = rag_system.retrieve(user_text, n_results=5)
        context = rag_system.format_context(chunks)
    except Exception:
        context = "(Context retrieval unavailable.)"
    
    if mode == "oracle":
        system_prompt = f"""You are a senior pharmaceutical MSL and market access expert specializing in Cardiovex (cardiovexaban) for secondary cardiovascular prevention. You are helping {rep_name}, a pharmaceutical sales representative, with questions about the product.

Answer questions directly and precisely, citing SHIELD-1, SHIELD-2, and the FDA label when relevant. Explain objection handling and market access strategies.

IMPORTANT: Write in plain text only. Do not use markdown formatting — no asterisks, no pound signs, no bullet dashes, no bold, no headers. Use plain sentences and paragraph breaks only.

Here is the relevant documentation:

{context}"""
    else:
        system_prompt = get_persona(persona_key, context_block=context)
    
    api_messages = [{"role": m["role"], "content": m["content"]} for m in history]
    api_messages.append({"role": "user", "content": user_text})
    
    try:
        assistant_text = call_claude(system_prompt, api_messages)
    except Exception as e:
        assistant_text = f"[API ERROR: {str(e)}]"
    
    new_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    
    return new_history, ""

# ---------------------------------------------------------------------------
# Export session as .txt
# ---------------------------------------------------------------------------

@app.callback(
    Output("export-download", "data"),
    Input("export-btn", "n_clicks"),
    State("chat-store", "data"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def export_session(n_clicks, history, state):
    if not history or not state:
        return no_update
    from datetime import date
    rep_name = state.get("rep_name", "Rep")
    mode = state.get("mode", "sim")
    persona_key = state.get("persona", "patricia")
    mode_label = "Oracle · Expert Q&A" if mode == "oracle" else f"Simulation · {PERSONAS.get(persona_key, {}).get('name', persona_key)}"
    lines = [
        "CARDIOVEX FIELD REP TRAINER — SESSION TRANSCRIPT",
        "⚠ SIMULATED DATA ONLY — CARDIOVEX IS A FICTIONAL PRODUCT ⚠",
        f"Rep: {rep_name}",
        f"Mode: {mode_label}",
        f"Date: {date.today().isoformat()}",
        "=" * 60,
        "",
    ]
    for msg in history:
        speaker = rep_name if msg["role"] == "user" else ("Expert" if mode == "oracle" else PERSONAS.get(persona_key, {}).get("name", "Persona"))
        lines.append(f"[{speaker.upper()}]")
        lines.append(msg["content"])
        lines.append("")
    content = "\n".join(lines)
    filename = f"cardiovex_session_{date.today().isoformat()}_{mode}.txt"
    return dcc.send_string(content, filename)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
