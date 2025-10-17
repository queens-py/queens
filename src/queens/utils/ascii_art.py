#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""ASCII art module."""

import logging

import pyfiglet

from queens.utils.printing import DEFAULT_OUTPUT_WIDTH

_logger = logging.getLogger(__name__)


def print_bmfia_acceleration(output_width: int = DEFAULT_OUTPUT_WIDTH) -> None:
    """Print BMFIA rocket.

    Args:
        output_width: Terminal output width
    """
    rocket = r"""
          !
          !
          ^
         / \
       / ___ \
      |=     =|
      |       |
      |-BMFIA-|
      |       |
      |       |
      |       |
      |       |
      |       |
      |       |
      |       |
     /| ##!## | \
   /  | ##!## |   \
 /    | ##!## |     \
|     / ^ | ^  \     |
|    /   (|)    \    |
|   /    (|)     \   |
|  /   ((   ))    \  |
| /     ((:))      \ |
|/      ((:))       \|
       ((   ))
        (( ))
         ( )
          .
          .
          .
    """
    print_centered_multiline_block(rocket, output_width)


def print_classification() -> None:
    """Print like a sir as the iterator is classification."""
    las = """                    ./@@@@@@@@
             *&@@@@@@@@@@@@@@@@(
        #@@@@@@@@@@@@@@@@@@@@@@@@
    /@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   ,@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.
     #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#       ,#@@@@@@@.
       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&@@@@@@@@@@@#
        /@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&.
          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
           ,@@@@@@@@@@@@@@@@@@.              *@
         ,@@@@@@@@@@@@@&        (@@@@@@@@@@@&  *%
  .#@@@@@@@@@@@@@@%    ,(###@      @*.%%/@@&&   /@
#@@@@@@@,   *@@&   *@ @*  ,@@@     /&      @     @     ,@&
 .          @@*  *@, @.  , ( ,(       %&@&,&@@@@@@@@@@@@@&
           #@& .@@@@@   #    ,(   /@@@@@@@@@@/    @
           @@,  #  /@  ,&  %,@  *@@@          @.  @
          ,@@*  &      *%%#    %@@   @& %,**   , %#
           @@@  *,          @@@@.     .         ,@
           ,@@@  #                             (@                ,*
            ,@@@,%                           ,@%              &@. &@.
              .@@@&                       .@@#              .@
                  &@@@@,             *@@@%% @@             %@
                      %@@,,#@    & /@@ /@%@ %@@@@@&/     /@/
                        .@   &%  .@@@@  @//, *@@.   %@@@@/
                         /@/   ., @  #@@@@  @/# @.
                        *@@@#   % %  #@(  @  &.  @/
                       %@  @@    .@% #@(   @ *.   @.
                      .@   &@       &,&@*  @#@,    @
                      @*  ,@@@@@@%##, #  ,(  @     #/

                            C L A S S ification
    """
    print_centered_multiline_block(las)


def print_crown(output_width: int = DEFAULT_OUTPUT_WIDTH) -> None:
    """Print crown.

    Args:
        output_width: Terminal output width
    """
    crown = r"""
        *
      * | *
     * \|/ *
* * * \|O|/ * * *
 \o\o\o|O|o/o/o/
 (<><><>O<><><>)
  '==========='
    """
    print_centered_multiline_block(crown, output_width)


def print_points_iterator(output_width: int = DEFAULT_OUTPUT_WIDTH) -> None:
    """Print points iterator.

    Args:
        output_width: Terminal output width
    """
    points = r"""
    @@@@@@@@@@@
  @@@        @@@                                           @@@@@@@@@@@
  @@@         @@@                                        @@@@       @@@@
  @@@        @@@                                         @@@         @@@
   @@@@@@,@@@@@                #@@@@@@@@@                @@@         @@@
    @@@@@@@@@@               @@@@      @@@@               @@@@@@@@@@@@@
      @@@@@@@                @@@         @@@                @@@@@@@@@@
       @@@@                  @@@        @@@                  @@@@@@@
        @@                    @@@      @@@                     @@@
                               @@@@@@@@@@                       @
                                @@@@@@@@
                                 @@@@@@
                                   @@
    """
    print_centered_multiline_block(points, output_width)


def print_banner(message: str = "QUEENS", output_width: int = DEFAULT_OUTPUT_WIDTH) -> None:
    """Print banner.

    Args:
        message: Message in banner
        output_width: Terminal output width
    """
    print_centered_multiline_block(pyfiglet.figlet_format(message, font="banner3-D"), output_width)


def print_centered_multiline_block(
    string: str | pyfiglet.FigletString, output_width: int = DEFAULT_OUTPUT_WIDTH
) -> None:
    """Print a multiline text in the center as a block.

    Args:
        string: String to be printed
        output_width: Terminal output width
    """
    lines = string.split("\n")
    max_line_width = max(len(line) for line in lines)
    if max_line_width % 2:
        output_width += 1
    for line in lines:
        _logger.info(line.ljust(max_line_width).center(output_width))


def print_centered_multiline(string: str, output_width: int = DEFAULT_OUTPUT_WIDTH) -> None:
    """Center every line of a multiline text.

    Args:
        string: String to be printed
        output_width: Terminal output width
    """
    lines = string.split("\n")
    for line in lines:
        _logger.info(line.strip().center(output_width))


def print_banner_and_description(output_width: int = DEFAULT_OUTPUT_WIDTH) -> None:
    """Print banner and the description.

    Args:
        output_width: Terminal output width
    """
    print_crown()
    print_banner()
    description = """
    A general purpose framework for Uncertainty Quantification,
    Physics-Informed Machine Learning, Bayesian Optimization,
    Inverse Problems and Simulation Analytics
    """
    print_centered_multiline(description, output_width)
