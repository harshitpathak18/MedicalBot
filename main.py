import time
import flet as ft
from medical import chatbot_with_iterative_refinement as medical_bot 

def chatbot_page(page: ft.Page):
    page.clean()
    page.title = "MedicalBot"
    page.foreground_decoration = None
    page.bgcolor = "#fef1e1"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    
    header = ft.Container(
        content=ft.Row(
            controls=[ft.Text("MedicalBot", color="#f7f4f3", size=20, weight=ft.FontWeight.BOLD)],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        ),
        padding=ft.padding.all(10),
        bgcolor="#a0430a",
        border_radius=ft.border_radius.only(top_left=10, top_right=10)
    )
    
    chat_column = ft.Column(expand=True, spacing=10, scroll=ft.ScrollMode.AUTO)

    def send_message(e):
        user_text = chat_input.value.strip()
        if user_text:
            chat_input.value = ""
            page.update()
            
            chat_column.controls.append(
                ft.Row([
                    ft.Container(
                        content=ft.Text(user_text, size=14, color="#fef1e1"),
                        padding=ft.padding.all(12),
                        bgcolor="#fc350b",
                        border_radius=20,
                        width=page.width * 0.4,
                    ),
                    ft.Icon(name=ft.Icons.PERSON, color="#fc350b", size=35),
                ], alignment=ft.MainAxisAlignment.END)
            )
            page.update()
            
            time.sleep(0.5)  # Short delay before bot response
            bot_response = medical_bot(user_text)
            # bot_response = "hello there, how are you doing man"
            chat_column.controls.append(
                ft.Row([
                    ft.Icon(name=ft.Icons.MEDICAL_SERVICES, color="#FADDBB", size=35),
                    ft.Container(
                        content=ft.Markdown(
                            bot_response,
                            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                            selectable=True,
                            width=page.width * 0.4,

                            md_style_sheet=ft.MarkdownStyleSheet(
                                # Text styles
                                a_text_style=ft.TextStyle(color="#9146FF", decoration="underline"),
                                p_text_style=ft.TextStyle(size=14, color="black"),
                                
                                # Heading styles
                                h1_text_style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD, color="#9146FF"),
                                h2_text_style=ft.TextStyle(size=16, weight=ft.FontWeight.BOLD, color="#9146FF"),
                                h3_text_style=ft.TextStyle(size=14, weight=ft.FontWeight.BOLD, color="black"),
                                
                                # List styles
                                list_bullet_text_style=ft.TextStyle(color="black"),
                                list_indent=20,
                                
                                # Spacing
                                block_spacing=10,
                                h1_padding=ft.padding.only(bottom=5),
                                h2_padding=ft.padding.only(bottom=5),
                                p_padding=ft.padding.only(bottom=8),
                            )      
                        ),
                        
                        padding=ft.padding.all(10),
                        bgcolor="#FADDBB",
                        border_radius=20,
                        width=page.width * 0.55,
                    )
                ], alignment=ft.MainAxisAlignment.START)
            )
            page.update()
    
    chat_input = ft.TextField(
        hint_text="Type your message...",
        expand=True,
        color='black',
        text_size=14,
        hint_style=ft.TextStyle(color="black"),
        bgcolor="#f2ca7f",
        border=ft.InputBorder.NONE,
        content_padding=ft.padding.all(10),
        on_submit=send_message
    )

    send_button = ft.IconButton(
        icon=ft.Icons.SEND,
        icon_color="#fcf7f8",
        bgcolor="#a0430a",
        on_click=send_message
    )

    input_row = ft.Container(
        content=ft.Row(
            controls=[chat_input, send_button],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER
        ),
        height=55,
        border_radius=30,
        bgcolor="#f2ca7f",
        padding=ft.padding.symmetric(horizontal=10),
        margin=ft.margin.only(bottom=10)
    )

    page.add(header, ft.Container(content=chat_column, expand=True, padding=10), input_row)



def main(page: ft.Page):
    page.window_width = 250
    page.window_height = 640
    page.window_resizable = False 
    page.title = "MedicalBot"
    page.foreground_decoration = ft.BoxDecoration(
        image=ft.DecorationImage(
            src="https://i.pinimg.com/474x/5c/b2/0d/5cb20d64f0bbb46eea7a0eca46bddfe7.jpg",
            fit=ft.ImageFit.COVER,
            opacity=0.2,
        ),
    )
    
    welcome_text = ft.Text(
        "Welcome to MedicalBot!", size=45, weight=ft.FontWeight.BOLD, color="#a0430a"
    )

    description_text = ft.Text(
        "Your AI-powered health assistant is here to help! "
        "Ask about symptoms, get health tips, and receive expert-backed advice. "
        "Covering over 2,000 topics, including diseases, disorders, conditions, tests, "
        "procedures, therapies, and other health-related issues. Stay informed, stay "
        "healthy – anytime, anywhere!,",
        size=20, color="white", text_align=ft.TextAlign.CENTER
    )

    
    start_button = ft.ElevatedButton(
        text="START",
        bgcolor="#a0430a",
        color="#f7f4f3",
        width=80, 
        height=40,
        on_click=lambda e: chatbot_page(page)
    )
    
    landing_page = ft.Column(
        controls=[
            welcome_text,
            ft.Container(content=description_text, padding=ft.padding.all(10)),
            ft.Container(content=start_button, padding=ft.padding.all(10))
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True
    )
    
    page.add(landing_page)

ft.app(target=main)
