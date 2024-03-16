#ref: https://stackoverflow.com/a/8873302/3811558
#     https://stackoverflow.com/a/22115860/3811558
import pygame

WIDTH = 1776
HEIGHT = 521
WHITE = (255,255,255) #RGB
GREEN = (0,255,0) #RGB

pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT),0,32)
pygame.display.set_caption("Indoor Positioning")
myimage = pygame.image.load("gui_test_background_img.jpg")
imagerect = myimage.get_rect()
timer = pygame.time.Clock()
left, top, radius = 50, 50, 10
rate_top  = 20
rate_left = 20
while 1:
    screen.fill(GREEN)
    screen.blit(myimage, imagerect)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    pygame.draw.circle(screen, GREEN, (left, top), radius)

    top_c  = top + rate_top
    left_c = left + rate_left
    if((top_c > HEIGHT) or (top_c < 0)):
        rate_top = -rate_top
        top = top; # unchanged
    else:
        rate_top = rate_top
        top = top_c; # unchanged
    if((left_c > WIDTH) or (left_c < 0)):
        rate_left = -rate_left
        left = left; # unchanged
    else:
        rate_left = rate_left
        left = left_c; # unchanged

    pygame.display.update()
    timer.tick(10) #60 times per second you can do the math for 17 ms
